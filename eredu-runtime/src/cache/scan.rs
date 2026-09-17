//! Shared ordered block/tail scan, independent of tensors and residency policy.
use eredu_nn::{BlockwiseAttentionOptions, BlockwiseAttentionPolicyError};
use std::mem::size_of;

/// Invalid scan coordinates or a policy rejected before any scan callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PagedScanError {
    /// The query span must end at the current nonnegative cache frontier.
    #[error("paged scan query differs from its cache frontier")]
    Frontier,
    /// Sliding width and retained prefix must have valid extents.
    #[error("paged scan window or prefix is invalid")]
    Window,
    /// The existing blockwise worker rejected the selected numerical policy.
    #[error(transparent)]
    Policy(#[from] BlockwiseAttentionPolicyError),
}

/// Descriptive scan coordinates. Source provenance, promotion, completion and
/// native permission remain with the concrete owner of each visited block.
#[derive(Clone, Copy, Debug)]
pub struct PagedScanPlan {
    query_start: i64,
    visible_start: i64,
    context_end: i64,
    prefix: i64,
    options: BlockwiseAttentionOptions,
}
impl PagedScanPlan {
    /// Derives the exact shared visibility window from the actual query span.
    pub fn new(
        context_end: i64,
        query_tokens: i32,
        window: Option<i32>,
        prefix: i64,
        options: BlockwiseAttentionOptions,
    ) -> Result<Self, PagedScanError> {
        if query_tokens <= 0 || context_end < i64::from(query_tokens) {
            return Err(PagedScanError::Frontier);
        }
        if window.is_some_and(|width| width <= 0) || prefix < 0 {
            return Err(PagedScanError::Window);
        }
        options.validate()?;
        let query_start = context_end - i64::from(query_tokens);
        let visible_start = window.map_or(0, |width| (query_start - (i64::from(width) - 1)).max(0));
        Ok(Self {
            query_start,
            visible_start,
            context_end,
            prefix,
            options,
        })
    }
    /// Actual first query position.
    pub const fn query_start(self) -> i64 {
        self.query_start
    }
    /// Earliest non-prefix key needed by any query in this span.
    pub const fn visible_start(self) -> i64 {
        self.visible_start
    }
    /// Uses the same interval predicate as the actual cache catalog selector.
    /// This method grants no source or block lease.
    pub fn selects(self, start: i64, end: i64) -> bool {
        super::source::interval_overlaps(
            start,
            end,
            self.visible_start,
            self.context_end,
            self.prefix,
        )
    }
    /// Fixed live traversal and callback frames, reused for each physical block.
    /// Actual source lists, leases and native work remain separately accounted.
    pub fn control_bytes<M: PagedScanMechanisms>() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<M>(),
            size_of::<&mut M>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<usize>(),
            size_of::<M::Cursor>(),
            size_of::<M::Block>(),
            size_of::<Result<M::Cursor, M::Error>>(),
            size_of::<Result<Option<M::Block>, M::Error>>(),
            size_of::<Result<(), M::Error>>(),
            size_of::<Result<M::Output, M::Error>>(),
            size_of::<Result<Self, PagedScanError>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Preserves the ordinary scan order: ordered sealed blocks are submitted
    /// while their lease remains live, followed by the independent partial tail.
    /// The cursor remains live through the tail and is dropped before finishing.
    pub fn run<M: PagedScanMechanisms>(self, mechanism: &mut M) -> Result<M::Output, M::Error> {
        for pass in 0..self.options.passes() {
            mechanism.begin_pass(pass)?;
            let mut cursor = mechanism.open_blocks()?;
            while let Some(block) = mechanism.next_block(&mut cursor)? {
                mechanism.consume_block(&block)?;
                mechanism.submit()?;
                drop(block);
            }
            mechanism.consume_tail()?;
        }
        mechanism.finish()
    }
}

/// Concrete operations under the shared scan traversal. Implementations retain
/// their existing source, promotion, numerical worker and completion contracts.
pub trait PagedScanMechanisms {
    /// Actual ordered block cursor or a symbolic index into retained metadata.
    type Cursor;
    /// Actual block lease or a validated symbolic source index.
    type Block;
    /// Result of the existing attention worker.
    type Output;
    /// The original error type, including any retained failure custody.
    type Error;
    /// Starts normalization, or transitions to the rounded value pass.
    fn begin_pass(&mut self, pass: usize) -> Result<(), Self::Error>;
    /// Opens the same exact source sequence for this numerical pass.
    fn open_blocks(&mut self) -> Result<Self::Cursor, Self::Error>;
    /// Acquires the next actual selected block.
    fn next_block(&mut self, cursor: &mut Self::Cursor)
    -> Result<Option<Self::Block>, Self::Error>;
    /// Consumes the block through the shared numerical accumulator.
    fn consume_block(&mut self, block: &Self::Block) -> Result<(), Self::Error>;
    /// Submits work while the current sealed block's lease remains live.
    fn submit(&mut self) -> Result<(), Self::Error>;
    /// Consumes the independent partial tail without an extra sealed submission.
    fn consume_tail(&mut self) -> Result<(), Self::Error>;
    /// Finishes and settles before discarding history or publishing telemetry.
    fn finish(&mut self) -> Result<Self::Output, Self::Error>;
}

#[cfg(test)]
mod tests;
