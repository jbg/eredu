//! Counted byte-preserving recipe mappings over borrowed child coordinates.
use super::{EncodedRange, RecipeError, children::ChildMappings, overflow};
use std::{alloc::Layout, ops::Range};

/// Owned byte coordinates for one encoded recipe node. Source coordinates refer
/// to the original concatenated batch; destination coordinates are contiguous.
/// No payload or source owner is copied into this mapping.
#[derive(Debug)]
pub struct EncodedRecipeMapping {
    pub(super) ranges: Vec<EncodedRange>,
    pub(super) length: usize,
}

pub(super) enum EncodedRecipeMappingInput<'a> {
    Source(Range<usize>),
    Selected {
        input: &'a EncodedRecipeMapping,
        ranges: &'a [Range<usize>],
    },
    Interleaved {
        children: Children<'a>,
        outer: usize,
    },
}
pub(super) enum Children<'a> {
    Borrowed(&'a [&'a EncodedRecipeMapping], &'a [usize]),
    Constructed(&'a dyn ChildMappings),
}
impl Children<'_> {
    fn len(&self) -> Result<usize, RecipeError> {
        match self {
            Self::Borrowed(rows, chunks) if rows.len() == chunks.len() => Ok(rows.len()),
            Self::Borrowed(..) => Err(overflow()),
            Self::Constructed(rows) => Ok(rows.len()),
        }
    }
    fn get(&self, index: usize) -> (&EncodedRecipeMapping, usize) {
        match self {
            Self::Borrowed(rows, chunks) => (rows[index], chunks[index]),
            Self::Constructed(rows) => rows.get(index),
        }
    }
}

impl EncodedRecipeMapping {
    pub(crate) fn encoded_ranges(&self) -> &[EncodedRange] {
        &self.ranges
    }

    /// Total encoded destination length, without reading payloads.
    pub fn byte_len(&self) -> usize {
        self.length
    }
    /// Source and destination coordinates in output order. Adjacent source
    /// ranges are coalesced. Returned ranges contain only scalar offsets.
    pub fn ranges(&self) -> impl ExactSizeIterator<Item = (Range<usize>, Range<usize>)> + '_ {
        self.ranges
            .iter()
            .map(|row| (row.source.clone(), row.destination.clone()))
    }
}

impl EncodedRecipeMappingInput<'_> {
    fn write(&self, writer: &mut Writer<'_>) -> Result<(), RecipeError> {
        match self {
            Self::Source(range) => writer.push(range.clone())?,
            Self::Selected { input, ranges } => {
                for range in *ranges {
                    writer.append_slice(input, range.clone())?;
                }
            }
            Self::Interleaved { children, outer } => {
                let count = children.len()?;
                for child_index in 0..count {
                    let (child, chunk) = children.get(child_index);
                    if outer.checked_mul(chunk).ok_or_else(overflow)? != child.length {
                        return Err(overflow());
                    }
                }
                for index in 0..*outer {
                    for child_index in 0..count {
                        let (child, chunk) = children.get(child_index);
                        let start = index.checked_mul(chunk).ok_or_else(overflow)?;
                        writer.append_slice(
                            child,
                            start..start.checked_add(chunk).ok_or_else(overflow)?,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Counted construction over immutable borrowed child mappings. Counting and
/// filling use the same traversal; construction owns its output independently
/// of all input mappings. Caller admission and child-array storage are separate.
pub struct EncodedRecipeMappingPlan<'a> {
    input: EncodedRecipeMappingInput<'a>,
    count: usize,
    length: usize,
    layout: Layout,
}
impl<'a> EncodedRecipeMappingPlan<'a> {
    /// Plan one source interval in the concatenated original batch.
    pub fn source(range: Range<usize>) -> Result<Self, RecipeError> {
        Self::new(EncodedRecipeMappingInput::Source(range))
    }
    /// Select output intervals from an existing mapping, preserving repetitions
    /// and selection order. The interval slice remains a borrowed prerequisite.
    pub fn selected(
        input: &'a EncodedRecipeMapping,
        ranges: &'a [Range<usize>],
    ) -> Result<Self, RecipeError> {
        Self::new(EncodedRecipeMappingInput::Selected { input, ranges })
    }
    /// Interleave equally many chunks from borrowed child mappings. Each child
    /// contributes `chunks[index]` bytes to each of `outer` output rows.
    pub fn interleaved(
        children: &'a [&'a EncodedRecipeMapping],
        chunks: &'a [usize],
        outer: usize,
    ) -> Result<Self, RecipeError> {
        Self::new(EncodedRecipeMappingInput::Interleaved {
            children: Children::Borrowed(children, chunks),
            outer,
        })
    }
    /// Exact requested range backing plus fixed constructor/result controls.
    /// Borrowed child arrays, allocator overhead and machine stack are excluded.
    pub fn required_bytes(&self) -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<EncodedRecipeMapping>(),
            std::mem::size_of::<RecipeError>(),
            std::mem::size_of::<Writer<'static>>(),
            std::mem::size_of::<Result<(), std::collections::TryReserveError>>(),
            std::mem::size_of::<Result<EncodedRecipeMapping, RecipeError>>(),
        ]
        .into_iter()
        .try_fold(self.layout.size(), usize::checked_add)
    }
    pub(super) fn new(input: EncodedRecipeMappingInput<'a>) -> Result<Self, RecipeError> {
        let mut writer = Writer {
            ranges: Destination::Count {
                count: 0,
                last: None,
            },
            length: 0,
        };
        input.write(&mut writer)?;
        let count = writer.ranges.len();
        let layout = Layout::array::<EncodedRange>(count).map_err(|_| overflow())?;
        Ok(Self {
            input,
            count,
            length: writer.length,
            layout,
        })
    }
    fn fill_into(&self, destination: &mut [EncodedRange]) -> Result<(), RecipeError> {
        if destination.len() != self.layout.size() / std::mem::size_of::<EncodedRange>() {
            return Err(overflow());
        }
        let mut writer = Writer {
            ranges: Destination::Fill {
                values: destination,
                used: 0,
            },
            length: 0,
        };
        self.input.write(&mut writer)?;
        if writer.ranges.len() != self.count || writer.length != self.length {
            return Err(overflow());
        }
        Ok(())
    }
    /// Allocate one exact requested range vector and fill it with the shared
    /// traversal. Allocation failure remains a typed cause; partial storage
    /// retires synchronously, without publishing a source or native alias.
    pub fn build(self) -> Result<EncodedRecipeMapping, RecipeError> {
        let mut ranges = Vec::new();
        ranges
            .try_reserve_exact(self.count)
            .map_err(RecipeError::ProjectionReserve)?;
        ranges.resize(
            self.count,
            EncodedRange {
                source: 0..0,
                destination: 0..0,
            },
        );
        self.fill_into(&mut ranges)?;
        Ok(EncodedRecipeMapping {
            ranges,
            length: self.length,
        })
    }
}

enum Destination<'a> {
    Count {
        count: usize,
        last: Option<EncodedRange>,
    },
    Fill {
        values: &'a mut [EncodedRange],
        used: usize,
    },
}
impl Destination<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Count { count, .. } => *count,
            Self::Fill { used, .. } => *used,
        }
    }
    fn last_mut(&mut self) -> Option<&mut EncodedRange> {
        match self {
            Self::Count { last, .. } => last.as_mut(),
            Self::Fill { values, used } => {
                used.checked_sub(1).and_then(|index| values.get_mut(index))
            }
        }
    }
    fn push(&mut self, row: EncodedRange) -> Result<(), RecipeError> {
        match self {
            Self::Count { count, last } => {
                *count = count.checked_add(1).ok_or_else(overflow)?;
                *last = Some(row);
            }
            Self::Fill { values, used } => {
                *values.get_mut(*used).ok_or_else(overflow)? = row;
                *used += 1;
            }
        }
        Ok(())
    }
}
struct Writer<'a> {
    ranges: Destination<'a>,
    length: usize,
}
impl Writer<'_> {
    fn push(&mut self, source: Range<usize>) -> Result<(), RecipeError> {
        let length = source.end.checked_sub(source.start).ok_or_else(overflow)?;
        if length == 0 {
            return Ok(());
        }
        let end = self.length.checked_add(length).ok_or_else(overflow)?;
        if let Some(last) = self
            .ranges
            .last_mut()
            .filter(|last| last.source.end == source.start)
        {
            last.source.end = source.end;
            last.destination.end = end;
        } else {
            self.ranges.push(EncodedRange {
                source,
                destination: self.length..end,
            })?;
        }
        self.length = end;
        Ok(())
    }
    fn append_slice(
        &mut self,
        input: &EncodedRecipeMapping,
        selected: Range<usize>,
    ) -> Result<(), RecipeError> {
        if selected.start > selected.end || selected.end > input.length {
            return Err(overflow());
        }
        let first = input
            .ranges
            .partition_point(|row| row.destination.end <= selected.start);
        for row in &input.ranges[first..] {
            if row.destination.start >= selected.end {
                break;
            }
            let start = row.destination.start.max(selected.start);
            let end = row.destination.end.min(selected.end);
            if start < end {
                let source_start = row
                    .source
                    .start
                    .checked_add(start - row.destination.start)
                    .ok_or_else(overflow)?;
                self.push(
                    source_start..source_start.checked_add(end - start).ok_or_else(overflow)?,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
