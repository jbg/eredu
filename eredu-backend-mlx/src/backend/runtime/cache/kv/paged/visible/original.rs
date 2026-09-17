//! Original source adapter for the ordinary local-window traversal.
use super::*;
use crate::backend::nn::workspace::{OriginalPagedAppendClaim, OriginalPagedVisibleClaim};
use std::mem::size_of;

pub(super) fn update(
    cache: &mut PagedKeyValueCache,
    keys: Array,
    values: Array,
    window: i32,
    stream: &Stream,
) -> Result<(Array, Array), Exception> {
    cache.with_original_append(keys, values, stream, |cache, claim, keys, values| {
        let plan =
            PagedLocalUpdatePlan::new(cache.offset, window, keys.dim(-2)).map_err(|cause| {
                claim.error(
                    crate::backend::runtime::cache::residency::CacheSourceError::Visible(cause),
                )
            })?;
        plan.run(&mut Local {
            cache,
            claim,
            stream,
            input: Some((keys, values)),
        })
    })
}
struct Local<'a, 'source> {
    cache: &'a mut PagedKeyValueCache,
    claim: &'a mut OriginalPagedAppendClaim<'source>,
    stream: &'a Stream,
    input: Option<(Array, Array)>,
}
impl PagedLocalUpdateMechanisms for Local<'_, '_> {
    type Pair = (Array, Array);
    type Error = Exception;
    fn visible(&mut self, plan: PagedVisiblePlan) -> Result<Self::Pair, Exception> {
        let tail = self
            .cache
            .tail_keys
            .as_ref()
            .zip(self.cache.tail_values.as_ref())
            .map(|(keys, values)| [keys, values]);
        self.claim.with_visible(plan, tail, |claim| {
            plan.run(&mut Visible {
                claim,
                stream: self.stream,
            })
        })
    }
    fn join_input(
        &mut self,
        (past_keys, past_values): Self::Pair,
    ) -> Result<Self::Pair, Exception> {
        let (keys, values) = self.input.as_ref().expect("input precedes append");
        let joined_keys = concatenate_axis(&[&past_keys, keys], -2, self.stream);
        drop(past_keys);
        let joined_keys = self.claim.retain_visible(joined_keys?)?;
        let joined_values = concatenate_axis(&[&past_values, values], -2, self.stream);
        drop(past_values);
        let joined_values = self.claim.retain_visible(joined_values?)?;
        Ok((joined_keys, joined_values))
    }
    fn copy_input(&mut self) -> Result<Self::Pair, Exception> {
        let (keys, values) = self.input.as_ref().expect("input precedes append");
        let keys = self.claim.retain_visible(keys.try_clone_handle()?)?;
        let values = self.claim.retain_visible(values.try_clone_handle()?)?;
        Ok((keys, values))
    }
    fn prepare_append(&mut self, visible: &Self::Pair) -> Result<(), Exception> {
        self.claim
            .complete_visible([&visible.0, &visible.1], self.stream)
    }
    fn append(&mut self) -> Result<(), Exception> {
        let (keys, values) = self.input.take().expect("one append");
        self.cache
            .append_claimed(self.claim, keys, values, false, self.stream)
    }
}
struct Visible<'a, 'target, 'source> {
    claim: &'a mut OriginalPagedVisibleClaim<'target, 'source>,
    stream: &'a Stream,
}
impl PagedVisibleMechanisms for Visible<'_, '_, '_> {
    type Cursor = ();
    type Block = usize;
    type Output = (Array, Array);
    type Error = Exception;
    fn error(&self, cause: PagedVisibleError) -> Exception {
        self.claim.error(cause)
    }
    fn open_blocks(&mut self, plan: PagedVisiblePlan) -> Result<(), Exception> {
        self.claim.validate_plan(plan)
    }
    fn next_block(&mut self, _: &mut ()) -> Result<Option<usize>, Exception> {
        self.claim.acquire_next(self.stream)
    }
    fn block_range(&self, block: &usize) -> (i64, i64) {
        self.claim.block_range(*block)
    }
    fn push_block(&mut self, block: &usize, range: Range<i32>) -> Result<(), Exception> {
        self.claim.push_block(*block, range, self.stream)
    }
    fn submit_block(&mut self) -> Result<(), Exception> {
        self.claim.submit_block(self.stream)
    }
    fn tail_range(&self) -> Option<(i64, i64)> {
        self.claim.tail_range()
    }
    fn push_tail(&mut self, range: Range<i32>) -> Result<(), Exception> {
        self.claim.push_tail(range, self.stream)
    }
    fn finish(&mut self, tokens: i64) -> Result<Self::Output, Exception> {
        self.claim.finish(tokens, self.stream)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(&mut PagedKeyValueCache, Array, Array, i32, &Stream)>(),
        size_of::<(
            &mut PagedKeyValueCache,
            &mut OriginalPagedAppendClaim<'_>,
            Array,
            Array,
            &Stream,
            i32,
        )>(),
        size_of::<(&mut Local<'_, '_>, PagedVisiblePlan, Option<[&Array; 2]>)>(),
        size_of::<(
            &mut Visible<'_, '_, '_>,
            &mut (),
            Option<usize>,
            Range<i32>,
            i64,
        )>(),
        size_of::<(
            &mut Local<'_, '_>,
            (Array, Array),
            (&Array, &Array),
            Result<Array, Exception>,
        )>(),
        size_of::<Result<(Array, Array), Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<[&Array; 2]>(),
        PagedLocalUpdatePlan::control_bytes::<Local<'_, '_>>()?,
        PagedVisiblePlan::control_bytes::<Visible<'_, '_, '_>>()?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
