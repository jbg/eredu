//! Counted byte-preserving recipe mappings over borrowed child coordinates.
use super::{EncodedRange, RecipeError, overflow};
use std::{alloc::Layout, ops::Range};

pub(super) struct Mapping {
    pub(super) ranges: Vec<EncodedRange>,
    pub(super) length: usize,
}

pub(super) enum MappingInput<'a> {
    Source(Range<usize>),
    Selected {
        input: &'a Mapping,
        ranges: &'a [Range<usize>],
    },
    Interleaved {
        children: &'a [Mapping],
        chunks: &'a [usize],
        outer: usize,
    },
}
impl MappingInput<'_> {
    fn write(&self, writer: &mut Writer<'_>) -> Result<(), RecipeError> {
        match self {
            Self::Source(range) => writer.push(range.clone())?,
            Self::Selected { input, ranges } => {
                for range in *ranges {
                    writer.append_slice(input, range.clone())?;
                }
            }
            Self::Interleaved {
                children,
                chunks,
                outer,
            } => {
                if children.len() != chunks.len() {
                    return Err(overflow());
                }
                for (child, chunk) in children.iter().zip(chunks.iter().copied()) {
                    if outer.checked_mul(chunk).ok_or_else(overflow)? != child.length {
                        return Err(overflow());
                    }
                }
                for index in 0..*outer {
                    for (child, chunk) in children.iter().zip(chunks.iter().copied()) {
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

pub(super) struct MappingPlan<'a> {
    input: MappingInput<'a>,
    count: usize,
    length: usize,
    layout: Layout,
}
impl<'a> MappingPlan<'a> {
    pub(super) fn new(input: MappingInput<'a>) -> Result<Self, RecipeError> {
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
    pub(super) fn build(self) -> Result<Mapping, RecipeError> {
        let mut ranges = vec![
            EncodedRange {
                source: 0..0,
                destination: 0..0
            };
            self.count
        ];
        self.fill_into(&mut ranges)?;
        Ok(Mapping {
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
    fn append_slice(&mut self, input: &Mapping, selected: Range<usize>) -> Result<(), RecipeError> {
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
