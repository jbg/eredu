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
            let mut projected = Vec::<ReadSpan>::new();
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
                            u64::try_from(start - span.destination.start).map_err(|_| invalid())?,
                        )
                        .ok_or_else(invalid)?;
                    let file_end = file_start
                        .checked_add(u64::try_from(length).map_err(|_| invalid())?)
                        .filter(|end| *end <= span.source.end)
                        .ok_or_else(invalid)?;
                    let destination_start = row
                        .destination
                        .start
                        .checked_add(start - row.source.start)
                        .ok_or_else(invalid)?;
                    let destination_end =
                        destination_start.checked_add(length).ok_or_else(invalid)?;
                    covered = covered.checked_add(length).ok_or_else(invalid)?;
                    projected.push(ReadSpan {
                        source: file_start..file_end,
                        destination: destination_start..destination_end,
                    });
                }
            }
            projected.sort_unstable_by_key(|span| span.source.start);
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
