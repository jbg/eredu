//! Shared visible-window ordering with actual native source leases.
mod original;
use super::*;
use crate::backend::runtime::cache::residency::{CacheBlockLease, CacheBlockPrefetch};
use eredu_runtime::cache::{
    PagedLocalUpdateMechanisms, PagedLocalUpdatePlan, PagedVisibleError, PagedVisibleMechanisms,
    PagedVisiblePlan,
};
use std::ops::Range;

pub(super) fn contiguous_visible(
    cache: &PagedKeyValueCache,
    start: i64,
    end: i64,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    let plan = PagedVisiblePlan::new(start, end).map_err(Exception::from_source)?;
    plan.run(&mut Visible {
        cache,
        stream,
        keys: Vec::new(),
        values: Vec::new(),
    })
}
struct Visible<'a> {
    cache: &'a PagedKeyValueCache,
    stream: &'a Stream,
    keys: Vec<Array>,
    values: Vec<Array>,
}
impl PagedVisibleMechanisms for Visible<'_> {
    type Cursor = CacheBlockPrefetch;
    type Block = CacheBlockLease;
    type Output = (Array, Array);
    type Error = Exception;
    fn error(&self, cause: PagedVisibleError) -> Exception {
        Exception::from_source(cause)
    }
    fn open_blocks(&mut self, plan: PagedVisiblePlan) -> Result<Self::Cursor, Exception> {
        let range = plan.range();
        let ids = self
            .cache
            .manager
            .layer_block_ids(
                self.cache.global_layer,
                CacheRepresentation::KeyValue,
                range.start,
                range.end,
                0,
            )
            .map_err(cache_residency_exception)?;
        self.cache
            .manager
            .prefetch_blocks(ids, self.stream)
            .map_err(cache_residency_exception)
    }
    fn next_block(&mut self, cursor: &mut Self::Cursor) -> Result<Option<Self::Block>, Exception> {
        cursor.next_block().map_err(cache_residency_exception)
    }
    fn block_range(&self, block: &Self::Block) -> (i64, i64) {
        (block.id().start, block.id().end)
    }
    fn push_block(&mut self, block: &Self::Block, range: Range<i32>) -> Result<(), Exception> {
        match block.arrays() {
            CacheBlockArrays::KeyValue { keys, values } => {
                self.keys
                    .push(keys.try_index_device((.., .., range.clone(), ..), self.stream)?);
                self.values
                    .push(values.try_index_device((.., .., range, ..), self.stream)?);
                Ok(())
            }
            _ => Err(Exception::custom(
                "paged key/value cache found an incompatible block representation",
            )),
        }
    }
    fn tail_range(&self) -> Option<(i64, i64)> {
        self.cache
            .tail_keys
            .as_ref()
            .zip(self.cache.tail_values.as_ref())
            .map(|_| (self.cache.tail_start, self.cache.offset))
    }
    fn push_tail(&mut self, range: Range<i32>) -> Result<(), Exception> {
        let keys = self
            .cache
            .tail_keys
            .as_ref()
            .expect("validated independent tail");
        let values = self
            .cache
            .tail_values
            .as_ref()
            .expect("validated independent tail");
        self.keys
            .push(keys.try_index_device((.., .., range.clone(), ..), self.stream)?);
        self.values
            .push(values.try_index_device((.., .., range, ..), self.stream)?);
        Ok(())
    }
    fn finish(&mut self, tokens: i64) -> Result<(Array, Array), Exception> {
        let keys = self.keys.iter().collect::<Vec<_>>();
        let values = self.values.iter().collect::<Vec<_>>();
        if keys.is_empty() {
            return Err(Exception::from_source(PagedVisibleError::History));
        }
        let keys = if keys.len() == 1 {
            keys[0].clone()
        } else {
            concatenate_axis(&keys, -2, self.stream)?
        };
        let values = if values.len() == 1 {
            values[0].clone()
        } else {
            concatenate_axis(&values, -2, self.stream)?
        };
        if i64::from(keys.dim(-2)) != tokens || i64::from(values.dim(-2)) != tokens {
            return Err(Exception::from_source(PagedVisibleError::History));
        }
        Ok((keys, values))
    }
}
pub(super) fn update(
    cache: &mut PagedKeyValueCache,
    keys: Array,
    values: Array,
    window: i32,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    if safemlx::OriginalScopeObserver::try_current()?.is_some() {
        return original::update(cache, keys, values, window, stream);
    }
    let plan = PagedLocalUpdatePlan::new(cache.offset, window, keys.dim(-2))
        .map_err(Exception::from_source)?;
    plan.run(&mut Update {
        cache,
        stream,
        input: Some((keys, values)),
    })
}
struct Update<'a> {
    cache: &'a mut PagedKeyValueCache,
    stream: &'a Stream,
    input: Option<(Array, Array)>,
}
impl PagedLocalUpdateMechanisms for Update<'_> {
    type Pair = (Array, Array);
    type Error = Exception;
    fn visible(&mut self, plan: PagedVisiblePlan) -> Result<Self::Pair, Exception> {
        let range = plan.range();
        self.cache
            .contiguous_visible(range.start, range.end, self.stream)
    }
    fn join_input(
        &mut self,
        (past_keys, past_values): Self::Pair,
    ) -> Result<Self::Pair, Exception> {
        let (keys, values) = self.input.as_ref().expect("input precedes append");
        let joined_keys = concatenate_axis(&[&past_keys, keys], -2, self.stream);
        drop(past_keys);
        let joined_keys = joined_keys?;
        let joined_values = concatenate_axis(&[&past_values, values], -2, self.stream);
        drop(past_values);
        Ok((joined_keys, joined_values?))
    }
    fn copy_input(&mut self) -> Result<Self::Pair, Exception> {
        Ok(self.input.as_ref().expect("input precedes append").clone())
    }
    fn append(&mut self) -> Result<(), Exception> {
        let (keys, values) = self.input.take().expect("one append");
        self.cache
            .append_normalized(keys, values, false, self.stream)
    }
}

impl PagedKeyValueCache {
    pub(crate) fn original_visible_control_bytes() -> Option<usize> {
        original::control_bytes()
    }
}
