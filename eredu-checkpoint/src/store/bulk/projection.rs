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

fn invalid() -> StoreError {
    StoreError::Internal("encoded recipe range projection is inconsistent".into())
}

impl EncodedReadBatch {
    /// Consume the exact admission, changing only its finite range geometry.
    /// No file is opened, no payload is read and no source identity is replaced.
    pub(crate) fn project_ranges(
        mut self,
        ranges: &[EncodedRange],
        output_bytes: usize,
    ) -> Result<Self, StoreError> {
        let mut end = 0;
        for row in ranges {
            if row.source.start > row.source.end
                || row.source.end > self.byte_len
                || row.destination.start != end
                || row.destination.start > row.destination.end
                || row.destination.end > output_bytes
                || row.source.len() != row.destination.len()
            {
                return Err(invalid());
            }
            end = row.destination.end;
        }
        if end != output_bytes {
            return Err(invalid());
        }
        let mut covered = 0usize;
        for spans in self
            .shards
            .iter_mut()
            .map(|(_, source)| &mut source.spans)
            .chain(self.memory.iter_mut().map(|source| &mut source.spans))
        {
            // Original batch spans have disjoint source coordinates even when
            // a physical tensor occurs repeatedly. Source order is restored below.
            spans.sort_unstable_by_key(|span| span.destination.start);
            let plan = SpanProjectionPlan::new(spans, ranges).map_err(StoreError::from)?;
            let mut projected = vec![
                ReadSpan {
                    source: 0..0,
                    destination: 0..0
                };
                plan.count
            ];
            plan.fill_into(&mut projected).map_err(StoreError::from)?;
            covered = covered.checked_add(plan.covered).ok_or_else(invalid)?;
            *spans = projected;
        }
        // This also rejects a gap in the original admitted source coordinates.
        // Repeated source rows count independently: each owns distinct output.
        if covered != output_bytes {
            return Err(invalid());
        }
        self.shards.retain(|(_, shard)| !shard.spans.is_empty());
        self.memory.retain(|source| !source.spans.is_empty());
        self.byte_len = output_bytes;
        Ok(self)
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
#[derive(Clone, Copy, Debug)]
enum ProjectionError {
    Invalid,
    Layout,
    Destination { expected: usize, actual: usize },
}
impl From<ProjectionError> for StoreError {
    fn from(error: ProjectionError) -> Self {
        match error {
            ProjectionError::Invalid => invalid(),
            ProjectionError::Layout => {
                StoreError::Internal("encoded projection span layout overflow".into())
            }
            ProjectionError::Destination { expected, actual } => StoreError::Internal(format!(
                "encoded projection destination has {actual} spans; expected {expected}"
            )),
        }
    }
}
impl<'a> SpanProjectionPlan<'a> {
    fn new(spans: &'a [ReadSpan], ranges: &'a [EncodedRange]) -> Result<Self, ProjectionError> {
        let mut previous = 0;
        for span in spans {
            if span.source.start > span.source.end
                || span.destination.start > span.destination.end
                || span.destination.start < previous
                || u64::try_from(span.destination.len()).ok()
                    != Some(span.source.end - span.source.start)
            {
                return Err(ProjectionError::Invalid);
            }
            previous = span.destination.end;
        }
        let mut count = 0usize;
        let covered = project(spans, ranges, |_| {
            count = count.checked_add(1).ok_or(ProjectionError::Layout)?;
            Ok(())
        })?;
        let layout =
            std::alloc::Layout::array::<ReadSpan>(count).map_err(|_| ProjectionError::Layout)?;
        Ok(Self {
            spans,
            ranges,
            count,
            covered,
            layout,
        })
    }
    fn fill_into(&self, destination: &mut [ReadSpan]) -> Result<(), ProjectionError> {
        if destination.len() != self.layout.size() / std::mem::size_of::<ReadSpan>() {
            return Err(ProjectionError::Destination {
                expected: self.count,
                actual: destination.len(),
            });
        }
        let mut next = 0;
        project(self.spans, self.ranges, |span| {
            let slot = destination.get_mut(next).ok_or(ProjectionError::Invalid)?;
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
    mut emit: impl FnMut(ReadSpan) -> Result<(), ProjectionError>,
) -> Result<usize, ProjectionError> {
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
                        .map_err(|_| ProjectionError::Invalid)?,
                )
                .ok_or(ProjectionError::Invalid)?;
            let file_end = file_start
                .checked_add(u64::try_from(length).map_err(|_| ProjectionError::Invalid)?)
                .filter(|end| *end <= span.source.end)
                .ok_or(ProjectionError::Invalid)?;
            let destination_start = row
                .destination
                .start
                .checked_add(start - row.source.start)
                .ok_or(ProjectionError::Invalid)?;
            let destination_end = destination_start
                .checked_add(length)
                .ok_or(ProjectionError::Invalid)?;
            covered = covered
                .checked_add(length)
                .ok_or(ProjectionError::Invalid)?;
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
