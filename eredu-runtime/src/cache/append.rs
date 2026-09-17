//! One ordered block/tail append traversal for native and metadata mechanisms.
use std::{mem::size_of, ops::Range};

/// Invalid descriptive geometry, before any append callback has run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PagedAppendError {
    /// Block size and input span must both be positive.
    #[error("paged append requires positive block and input extents")]
    Extent,
    /// A partial tail must end at the exact current frontier.
    #[error("paged append tail differs from its current frontier")]
    Tail,
    /// The resulting absolute position does not fit the cache coordinate type.
    #[error("paged append position overflow")]
    Overflow,
}

/// Validated token partition. This contains no storage or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedAppendPlan {
    block_size: i32,
    input_len: i32,
    tail_len: i32,
    tail_start: i64,
    offset: i64,
    end: i64,
}
impl PagedAppendPlan {
    /// Validates an actual tail/frontier before any tensor or manager mutation.
    pub fn new(
        block_size: i32,
        input_len: i32,
        tail_len: i32,
        tail_start: i64,
        offset: i64,
    ) -> Result<Self, PagedAppendError> {
        if block_size <= 0 || input_len <= 0 {
            return Err(PagedAppendError::Extent);
        }
        if tail_len < 0
            || tail_len >= block_size
            || tail_start < 0
            || offset < 0
            || tail_start.checked_add(i64::from(tail_len)) != Some(offset)
        {
            return Err(PagedAppendError::Tail);
        }
        let end = offset
            .checked_add(i64::from(input_len))
            .ok_or(PagedAppendError::Overflow)?;
        Ok(Self {
            block_size,
            input_len,
            tail_len,
            tail_start,
            offset,
            end,
        })
    }

    /// Exact initial partial-tail start, length and absolute frontier.
    pub fn initial_frontier(&self) -> (i64, i32, i64) { (self.tail_start, self.tail_len, self.offset) }

    /// Exact count of newly sealed full blocks, excluding an unfinished suffix.
    pub fn sealed_blocks(&self) -> usize {
        ((i64::from(self.tail_len) + i64::from(self.input_len)) / i64::from(self.block_size))
            as usize
    }

    /// Exact successful tail/frontier after the same partition used by `run`.
    /// This is descriptive geometry, not proof that callbacks completed.
    pub fn resulting_tail(&self) -> (i64, i32, i64) {
        let tail_len = ((i64::from(self.tail_len) + i64::from(self.input_len))
            % i64::from(self.block_size)) as i32;
        (self.end - i64::from(tail_len), tail_len, self.end)
    }

    /// Shared exact partition used by the actual append driver and cold native
    /// publication preparation. Iteration owns only validated scalar geometry.
    pub fn steps(&self) -> PagedAppendSteps {
        PagedAppendSteps { plan: *self, input_start: 0, tail_len: self.tail_len, tail_start: self.tail_start }
    }

    /// Fixed live traversal frames; tensor/source allocations remain the adapter's
    /// own producers. The same frames are reused for each ordered input segment.
    pub fn control_bytes<M: PagedAppendMechanisms>() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<&mut M>(),
            size_of::<M>(),
            size_of::<PagedAppendSteps>(),
            size_of::<PagedAppendStep>(),
            size_of::<Option<PagedAppendStep>>(),
            size_of::<&mut PagedAppendSteps>(),
            size_of::<(i32, i32)>(),
            size_of::<Range<i32>>(),
            size_of::<M::Pair>(),
            size_of::<M::Pair>(),
            size_of::<Result<M::Pair, M::Error>>(),
            size_of::<Result<(), M::Error>>(),
            size_of::<Result<Self, PagedAppendError>>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    /// Runs the ordinary slice, join, account, publish and seal order. The caller
    /// keeps its original rollback/checkpoint transaction around this worker.
    pub fn run<M: PagedAppendMechanisms>(
        self,
        mechanism: &mut M,
        retain_for_attention: bool,
    ) -> Result<(), M::Error> {
        for step in self.steps() {
            let part = mechanism.slice_input(step.input())?;
            let candidate = mechanism.join_tail(part)?;
            mechanism.publish_tail(step.tail_start, step.tail_end, candidate)?;
            if step.seals {
                mechanism.seal_tail()?;
            }
        }
        mechanism.finish(self.end, retain_for_attention)
    }
}

/// One actual input segment and its resulting publication frontier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedAppendStep {
    input_start: i32,
    input_end: i32,
    tail_start: i64,
    tail_end: i64,
    seals: bool,
}
impl PagedAppendStep {
    /// Relative interval in this append's supplied tensor.
    pub fn input(self) -> Range<i32> { self.input_start..self.input_end }
    /// Absolute complete candidate tail interval, including any prior tail.
    pub fn tail(self) -> Range<i64> { self.tail_start..self.tail_end }
    /// Whether this candidate fills and seals the immutable block.
    pub fn seals(self) -> bool { self.seals }
}
/// Allocation-free ordered traversal of one already validated append plan.
#[derive(Clone, Debug)]
pub struct PagedAppendSteps {
    plan: PagedAppendPlan,
    input_start: i32,
    tail_len: i32,
    tail_start: i64,
}
impl Iterator for PagedAppendSteps {
    type Item = PagedAppendStep;
    fn next(&mut self) -> Option<Self::Item> {
        if self.input_start == self.plan.input_len { return None; }
        if self.tail_len == 0 { self.tail_start = self.plan.offset + i64::from(self.input_start); }
        let take = (self.plan.block_size - self.tail_len).min(self.plan.input_len - self.input_start);
        let input_end = self.input_start + take;
        self.tail_len += take;
        let step = PagedAppendStep {
            input_start: self.input_start, input_end,
            tail_start: self.tail_start, tail_end: self.tail_start + i64::from(self.tail_len),
            seals: self.tail_len == self.plan.block_size,
        };
        self.input_start = input_end;
        if step.seals { self.tail_len = 0; }
        Some(step)
    }
}
impl std::iter::FusedIterator for PagedAppendSteps {}

/// Concrete tensor and cache-account operations under the shared traversal.
/// Native implementations retain their original manager/rollback contract;
/// metadata implementations describe the same operations without native work.
pub trait PagedAppendMechanisms {
    /// Two aligned key/value arrays, including a key-only persistence sentinel.
    type Pair;
    /// The caller's original error type and source custody.
    type Error;
    /// Slices this exact input span along its sequence axis.
    fn slice_input(&mut self, input: Range<i32>) -> Result<Self::Pair, Self::Error>;
    /// Concatenates the previous partial tail with the sliced input, if present.
    fn join_tail(&mut self, part: Self::Pair) -> Result<Self::Pair, Self::Error>;
    /// Accounts a candidate before publishing it as the new partial tail.
    fn publish_tail(&mut self, start: i64, end: i64, pair: Self::Pair) -> Result<(), Self::Error>;
    /// Seals a complete tail using the same actual cache manager transition.
    fn seal_tail(&mut self) -> Result<(), Self::Error>;
    /// Publishes the final frontier, then discards history only if permitted.
    fn finish(&mut self, end: i64, retain_for_attention: bool) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_steps_share_partial_tail_seal_order_and_checked_frontiers() {
        let plan = PagedAppendPlan::new(4, 7, 2, 8, 10).unwrap();
        let steps: Vec<_> = plan.steps().map(|step| (step.input(), step.tail(), step.seals())).collect();
        assert_eq!(steps, [(0..2, 8..12, true), (2..6, 12..16, true), (6..7, 16..17, false)]);
        assert_eq!(plan.sealed_blocks(), 2);
        assert_eq!(plan.resulting_tail(), (16, 1, 17));
        #[derive(Default)] struct Calls(Vec<&'static str>);
        impl PagedAppendMechanisms for Calls {
            type Pair = (); type Error = ();
            fn slice_input(&mut self, _: Range<i32>) -> Result<(), ()> { self.0.push("slice"); Ok(()) }
            fn join_tail(&mut self, _: ()) -> Result<(), ()> { self.0.push("join"); Ok(()) }
            fn publish_tail(&mut self, _: i64, end: i64, _: ()) -> Result<(), ()> {
                self.0.push("publish"); if end == 16 { Err(()) } else { Ok(()) }
            }
            fn seal_tail(&mut self) -> Result<(), ()> { self.0.push("seal"); Ok(()) }
            fn finish(&mut self, _: i64, _: bool) -> Result<(), ()> { self.0.push("finish"); Ok(()) }
        }
        let mut calls = Calls::default();
        assert!(plan.run(&mut calls, false).is_err());
        assert_eq!(calls.0, ["slice", "join", "publish", "seal", "slice", "join", "publish"]);
        assert!(PagedAppendPlan::new(4, 1, 0, i64::MAX, i64::MAX).is_err());
        assert!(PagedAppendPlan::new(4, 1, 2, 8, 11).is_err());
    }
}
