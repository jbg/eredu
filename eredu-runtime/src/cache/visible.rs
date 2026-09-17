//! Shared transient visible-window assembly; retained history stays paged.
use std::{
    mem::{size_of, size_of_val},
    ops::Range,
};

/// A visible local window must have representable, ordered token coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PagedVisibleError {
    #[error("paged visible window has invalid coordinates")]
    Extent,
    #[error("paged visible history is empty or incomplete")]
    History,
}

/// Exact requested transient window. It carries no source or execution grant.
#[derive(Clone, Copy, Debug)]
pub struct PagedVisiblePlan {
    start: i64,
    end: i64,
}
impl PagedVisiblePlan {
    pub fn new(start: i64, end: i64) -> Result<Self, PagedVisibleError> {
        if start < 0 || end <= start || i32::try_from(end - start).is_err() {
            return Err(PagedVisibleError::Extent);
        }
        Ok(Self { start, end })
    }
    pub fn range(self) -> Range<i64> {
        self.start..self.end
    }
    pub fn selects(self, start: i64, end: i64) -> bool {
        super::source::interval_overlaps(start, end, self.start, self.end, 0)
    }
    /// Exact block-relative slice, shared by source and metadata consumers.
    pub fn clip(self, start: i64, end: i64) -> Result<Option<Range<i32>>, PagedVisibleError> {
        if start < 0 || end <= start {
            return Err(PagedVisibleError::Extent);
        }
        if !self.selects(start, end) {
            return Ok(None);
        }
        let first =
            i32::try_from(self.start.max(start) - start).map_err(|_| PagedVisibleError::Extent)?;
        let last =
            i32::try_from(self.end.min(end) - start).map_err(|_| PagedVisibleError::Extent)?;
        Ok(Some(first..last))
    }
    pub fn control_bytes<M: PagedVisibleMechanisms>() -> Option<usize> {
        let controls = [
            size_of::<Self>(),
            size_of::<M>(),
            size_of::<&mut M>(),
            size_of::<M::Cursor>(),
            size_of::<M::Block>(),
            size_of::<Range<i32>>(),
            size_of::<(i64, i64)>(),
            size_of::<Option<Range<i32>>>(),
            size_of::<Result<Option<M::Block>, M::Error>>(),
            size_of::<Result<M::Cursor, M::Error>>(),
            size_of::<Result<M::Output, M::Error>>(),
            size_of::<Result<(), M::Error>>(),
            size_of::<Result<Option<Range<i32>>, PagedVisibleError>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Preserves the ordinary order and lifetimes: slice each leased block,
    /// release that lease, slice the tail, then finish while the cursor lives.
    pub fn run<M: PagedVisibleMechanisms>(self, mechanism: &mut M) -> Result<M::Output, M::Error> {
        let mut cursor = mechanism.open_blocks(self)?;
        while let Some(block) = mechanism.next_block(&mut cursor)? {
            let (start, end) = mechanism.block_range(&block);
            if let Some(range) = self
                .clip(start, end)
                .map_err(|cause| mechanism.error(cause))?
            {
                mechanism.push_block(&block, range)?;
                mechanism.submit_block()?;
            }
            drop(block);
        }
        if let Some((start, end)) = mechanism.tail_range() {
            if let Some(range) = self
                .clip(start, end)
                .map_err(|cause| mechanism.error(cause))?
            {
                mechanism.push_tail(range)?;
            }
        }
        let result = mechanism.finish(self.end - self.start);
        drop(cursor);
        result
    }
}

pub trait PagedVisibleMechanisms {
    type Cursor;
    type Block;
    type Output;
    type Error;
    fn error(&self, cause: PagedVisibleError) -> Self::Error;
    fn open_blocks(&mut self, plan: PagedVisiblePlan) -> Result<Self::Cursor, Self::Error>;
    fn next_block(&mut self, cursor: &mut Self::Cursor)
    -> Result<Option<Self::Block>, Self::Error>;
    fn block_range(&self, block: &Self::Block) -> (i64, i64);
    fn push_block(&mut self, block: &Self::Block, range: Range<i32>) -> Result<(), Self::Error>;
    /// Qualified adapters may settle the just-produced slices while the exact
    /// page lease is still held. Ordinary implementations retain existing work.
    fn submit_block(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn tail_range(&self) -> Option<(i64, i64)>;
    fn push_tail(&mut self, range: Range<i32>) -> Result<(), Self::Error>;
    fn finish(&mut self, tokens: i64) -> Result<Self::Output, Self::Error>;
}

/// Shared order of local-cache readout and mutation. The previous visible
/// window is assembled before append can discard final-query-expired blocks.
#[derive(Clone, Copy, Debug)]
pub struct PagedLocalUpdatePlan {
    history: Option<PagedVisiblePlan>,
}
impl PagedLocalUpdatePlan {
    pub fn new(offset: i64, window: i32, input_tokens: i32) -> Result<Self, PagedVisibleError> {
        if offset < 0
            || window <= 0
            || input_tokens <= 0
            || offset.checked_add(i64::from(input_tokens)).is_none()
        {
            return Err(PagedVisibleError::Extent);
        }
        let start = (offset - (i64::from(window) - 1)).max(0);
        Ok(Self {
            history: (start < offset)
                .then(|| PagedVisiblePlan::new(start, offset))
                .transpose()?,
        })
    }
    /// Exact old-window interval, with no source or execution authority.
    pub fn history(self) -> Option<PagedVisiblePlan> {
        self.history
    }
    pub fn control_bytes<M: PagedLocalUpdateMechanisms>() -> Option<usize> {
        let controls = [
            size_of::<Self>(),
            size_of::<M>(),
            size_of::<&mut M>(),
            size_of::<M::Pair>(),
            size_of::<&M::Pair>(),
            size_of::<Result<M::Pair, M::Error>>(),
            size_of::<Result<(), M::Error>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    pub fn run<M: PagedLocalUpdateMechanisms>(
        self,
        mechanism: &mut M,
    ) -> Result<M::Pair, M::Error> {
        let visible = match self.history {
            Some(plan) => {
                let past = mechanism.visible(plan)?;
                mechanism.join_input(past)?
            }
            None => mechanism.copy_input()?,
        };
        mechanism.prepare_append(&visible)?;
        mechanism.append()?;
        Ok(visible)
    }
}
pub trait PagedLocalUpdateMechanisms {
    type Pair;
    type Error;
    fn visible(&mut self, plan: PagedVisiblePlan) -> Result<Self::Pair, Self::Error>;
    fn join_input(&mut self, past: Self::Pair) -> Result<Self::Pair, Self::Error>;
    fn copy_input(&mut self) -> Result<Self::Pair, Self::Error>;
    /// A source-qualified adapter may settle the actual visible pair before
    /// mutation can retire its old pages. Ordinary adapters retain their
    /// existing cursor/array ownership and need no extra boundary here.
    fn prepare_append(&mut self, _visible: &Self::Pair) -> Result<(), Self::Error> {
        Ok(())
    }
    fn append(&mut self) -> Result<(), Self::Error>;
}
