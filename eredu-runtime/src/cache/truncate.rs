//! Ordered truncation shared by native pagers and descriptive source traversal.
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

/// Invalid cache coordinates, before any source or tensor callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PagedTruncateError {
    /// The requested frontier is outside the current cache.
    #[error("paged truncation frontier is outside the current cache")]
    Frontier,
    /// The supplied crossing block does not contain the requested cut.
    #[error("paged truncation crossing block differs from the selected cut")]
    Crossing,
    /// A relative tensor coordinate does not fit its index type.
    #[error("paged truncation tensor coordinate overflow")]
    Overflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Traversal {
    Tail { retained: i32 },
    Catalog { crossing: Option<(i64, i64, i32)> },
    Checkpoint { saved_offset: i64 },
}

/// Validated descriptive geometry. A checkpoint traversal is not evidence that
/// a saved source belongs to the current pager; the adapter must retain and
/// authenticate both sources before its publication callback mutates storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedTruncatePlan {
    end: i64,
    traversal: Traversal,
}
impl PagedTruncatePlan {
    /// Selects the actual tail or immutable-block cut from a catalog inspection.
    pub fn new(
        offset: i64,
        tail_start: i64,
        end: i64,
        crossing: Option<Range<i64>>,
    ) -> Result<Self, PagedTruncateError> {
        if tail_start < 0 || tail_start > offset || end < 0 || end > offset {
            return Err(PagedTruncateError::Frontier);
        }
        let traversal = if end >= tail_start {
            if crossing.is_some() {
                return Err(PagedTruncateError::Crossing);
            }
            Traversal::Tail {
                retained: i32::try_from(end - tail_start)
                    .map_err(|_| PagedTruncateError::Overflow)?,
            }
        } else {
            let crossing = crossing
                .map(|range| {
                    if range.start < 0
                        || range.start >= end
                        || range.end <= end
                        || range.end > tail_start
                    {
                        return Err(PagedTruncateError::Crossing);
                    }
                    Ok((
                        range.start,
                        range.end,
                        i32::try_from(end - range.start)
                            .map_err(|_| PagedTruncateError::Overflow)?,
                    ))
                })
                .transpose()?;
            Traversal::Catalog { crossing }
        };
        Ok(Self { end, traversal })
    }

    /// Describes restoring an append-only checkpoint's sealed prefix and saved
    /// immutable tail aliases. No partial block is manufactured at saved_offset.
    pub fn checkpoint(
        offset: i64,
        tail_start: i64,
        saved_offset: i64,
        saved_tail_start: i64,
    ) -> Result<Self, PagedTruncateError> {
        if tail_start < 0
            || tail_start > offset
            || saved_tail_start < 0
            || saved_tail_start > saved_offset
            || saved_offset > offset
            || saved_tail_start > tail_start
        {
            return Err(PagedTruncateError::Frontier);
        }
        Ok(Self {
            end: saved_tail_start,
            traversal: Traversal::Checkpoint { saved_offset },
        })
    }

    /// Runs the selected ordered worker; all ownership and completion remain
    /// with the adapter. A failed callback stops before every subsequent step.
    pub fn run<M: PagedTruncateMechanisms>(self, mechanism: &mut M) -> Result<(), M::Error> {
        match self.traversal {
            Traversal::Tail { retained } => {
                let pair = mechanism.slice_tail(retained)?;
                mechanism.publish_tail(self.end, retained, pair)
            }
            Traversal::Catalog { crossing } => {
                let replacement = match crossing {
                    None => None,
                    Some((start, end, retained)) => {
                        let source = mechanism.acquire_block(start..end)?;
                        let pair = mechanism.slice_block(&source, retained)?;
                        mechanism.complete_slices(&pair)?;
                        let pair = mechanism.copy_slices(pair)?;
                        Some((source, pair))
                    }
                };
                mechanism.publish_catalog(self.end, replacement)
            }
            Traversal::Checkpoint { saved_offset } => {
                mechanism.restore_checkpoint(self.end, saved_offset)
            }
        }
    }

    /// Named fixed frames of this same traversal; source storage and tensor
    /// allocations are reported separately by the selected adapter.
    pub fn control_bytes<M: PagedTruncateMechanisms>() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<&mut M>(),
            size_of::<M>(),
            size_of::<M::Source>(),
            size_of::<M::Pair>(),
            size_of::<Option<(M::Source, M::Pair)>>(),
            size_of::<Range<i64>>(),
            size_of::<Result<M::Source, M::Error>>(),
            size_of::<Result<M::Pair, M::Error>>(),
            size_of::<Result<(), M::Error>>(),
            size_of::<Result<Self, PagedTruncateError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

/// Concrete catalog, native completion and saved-source operations used by the
/// shared traversal. Descriptive implementations record the same selected calls.
pub trait PagedTruncateMechanisms {
    /// Retained actual crossing block source.
    type Source;
    /// Native or descriptive key/value pair.
    type Pair;
    /// Concrete source or operation failure.
    type Error;
    /// Slices the current mutable tail.
    fn slice_tail(&mut self, retained: i32) -> Result<Self::Pair, Self::Error>;
    /// Publishes the selected tail and frontier.
    fn publish_tail(
        &mut self,
        end: i64,
        retained: i32,
        pair: Self::Pair,
    ) -> Result<(), Self::Error>;
    /// Acquires the exact inspected immutable block.
    fn acquire_block(&mut self, interval: Range<i64>) -> Result<Self::Source, Self::Error>;
    /// Slices both arrays from that retained source.
    fn slice_block(
        &mut self,
        source: &Self::Source,
        retained: i32,
    ) -> Result<Self::Pair, Self::Error>;
    /// Establishes completion before making independent copies.
    fn complete_slices(&mut self, pair: &Self::Pair) -> Result<(), Self::Error>;
    /// Applies the existing contiguous and independent-copy operations.
    fn copy_slices(&mut self, pair: Self::Pair) -> Result<Self::Pair, Self::Error>;
    /// Publishes the selected catalog removal or replacement.
    fn publish_catalog(
        &mut self,
        end: i64,
        replacement: Option<(Self::Source, Self::Pair)>,
    ) -> Result<(), Self::Error>;
    /// Authenticates and restores the actual saved source without a crossing copy.
    fn restore_checkpoint(&mut self, tail_start: i64, offset: i64) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Trace {
        calls: Vec<&'static str>,
        fail_completion: bool,
    }
    impl PagedTruncateMechanisms for Trace {
        type Source = Range<i64>;
        type Pair = i32;
        type Error = ();
        fn slice_tail(&mut self, n: i32) -> Result<i32, ()> {
            self.calls.push("tail slice");
            Ok(n)
        }
        fn publish_tail(&mut self, _: i64, _: i32, _: i32) -> Result<(), ()> {
            self.calls.push("tail publish");
            Ok(())
        }
        fn acquire_block(&mut self, r: Range<i64>) -> Result<Range<i64>, ()> {
            self.calls.push("acquire");
            Ok(r)
        }
        fn slice_block(&mut self, _: &Range<i64>, n: i32) -> Result<i32, ()> {
            self.calls.push("block slice");
            Ok(n)
        }
        fn complete_slices(&mut self, _: &i32) -> Result<(), ()> {
            self.calls.push("complete");
            if self.fail_completion {
                Err(())
            } else {
                Ok(())
            }
        }
        fn copy_slices(&mut self, n: i32) -> Result<i32, ()> {
            self.calls.push("copy");
            Ok(n)
        }
        fn publish_catalog(&mut self, _: i64, _: Option<(Range<i64>, i32)>) -> Result<(), ()> {
            self.calls.push("catalog publish");
            Ok(())
        }
        fn restore_checkpoint(&mut self, _: i64, _: i64) -> Result<(), ()> {
            self.calls.push("saved source");
            Ok(())
        }
    }
    #[test]
    fn truncation_preserves_completion_before_copy_and_stops_on_failure() {
        let plan = PagedTruncatePlan::new(10, 8, 5, Some(4..8)).unwrap();
        let mut trace = Trace::default();
        plan.run(&mut trace).unwrap();
        assert_eq!(
            trace.calls,
            [
                "acquire",
                "block slice",
                "complete",
                "copy",
                "catalog publish"
            ]
        );
        let mut trace = Trace {
            fail_completion: true,
            ..Trace::default()
        };
        assert!(plan.run(&mut trace).is_err());
        assert_eq!(trace.calls, ["acquire", "block slice", "complete"]);
    }
    #[test]
    fn checkpoint_restores_saved_tail_without_truncating_inside_its_sealed_replacement() {
        let mut trace = Trace::default();
        PagedTruncatePlan::checkpoint(10, 8, 7, 4)
            .unwrap()
            .run(&mut trace)
            .unwrap();
        assert_eq!(trace.calls, ["saved source"]);
        assert_eq!(
            PagedTruncatePlan::new(10, 8, 5, Some(0..4)),
            Err(PagedTruncateError::Crossing)
        );
        assert_eq!(
            PagedTruncatePlan::checkpoint(6, 4, 7, 4),
            Err(PagedTruncateError::Frontier)
        );
        let mut trace = Trace::default();
        PagedTruncatePlan::new(10, 8, 9, None)
            .unwrap()
            .run(&mut trace)
            .unwrap();
        assert_eq!(trace.calls, ["tail slice", "tail publish"]);
    }
}
