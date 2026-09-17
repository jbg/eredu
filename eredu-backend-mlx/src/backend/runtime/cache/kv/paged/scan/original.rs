//! Actual source/lease adapter for the shared ordered scan and accumulator.
use super::*;
use crate::backend::nn::workspace::{
    OriginalPagedAppendClaim, OriginalPagedScanClaim, PagedScanInput, ProjectedPagedSources,
};
use crate::backend::runtime::cache::residency::CacheSourceError;
use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions};
use safemlx::OriginalScopeObserver;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    cache: &mut PagedKeyValueCache,
    queries: &Array,
    scale: f32,
    mask: Option<&Array>,
    sinks: Option<&Array>,
    softcap: Option<f32>,
    arithmetic: AttentionArithmetic,
    relative: Option<&crate::backend::nn::relative_attention::RelativeAttentionKernel>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let observer = OriginalScopeObserver::require_current()?;
    let fail = |cause| OriginalPagedAppendClaim::input_error(&observer, cause);
    if queries.ndim() != 4 {
        return Err(fail(CacheSourceError::Geometry));
    }
    if relative.is_some() {
        return Err(fail(CacheSourceError::RelativeScan));
    }
    let options = BlockwiseAttentionOptions {
        arithmetic,
        softcap,
    };
    let plan = PagedScanPlan::new(
        cache.offset,
        queries.dim(-2),
        cache.sliding_window,
        i64::from(cache.prefix_tokens),
        options,
    )
    .map_err(|cause| fail(CacheSourceError::Scan(cause)))?;
    let manager = cache.manager.clone();
    let actual = PagedScanInput {
        manager: &manager,
        global_layer: cache.global_layer,
        rank: cache.rank,
        queries: [
            queries.dim(0),
            queries.dim(1),
            queries.dim(2),
            queries.dim(3),
        ],
        dtype: queries.dtype(),
        block_size: cache.manager.options().block_size_tokens(),
        offset: cache.offset,
        tail_start: cache.tail_start,
        window: cache.sliding_window,
        prefix: cache.prefix_tokens,
        options,
    };
    ProjectedPagedSources::with_current_scan(actual, |claim| {
        claim.validate_readout(mask, sinks)?;
        claim.bind_sources()?;
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            queries,
            scale,
            mask,
            plan.query_start(),
            cache.sliding_window,
            i64::from(cache.prefix_tokens),
            sinks,
            cache.offset,
            stream,
        )?;
        accumulator.set_options(options)?;
        *claim.accumulator() = Some(accumulator);
        plan.run(&mut OriginalScan {
            claim,
            queries,
            stream,
            scanned_blocks: 0,
            scanned_bytes: 0,
            scratch: 0,
        })
    })
    .map(Some)
}
struct OriginalScan<'a, 'source> {
    claim: &'a mut OriginalPagedScanClaim<'source>,
    queries: &'a Array,
    stream: &'a Stream,
    scanned_blocks: u64,
    scanned_bytes: u64,
    scratch: u64,
}
impl OriginalScan<'_, '_> {
    fn account(&mut self, start: i64, end: i64, bytes: u64) {
        account_block(
            self.queries,
            start,
            end,
            bytes,
            &mut self.scanned_blocks,
            &mut self.scanned_bytes,
            &mut self.scratch,
        );
    }
}
impl PagedScanMechanisms for OriginalScan<'_, '_> {
    type Cursor = usize;
    type Block = usize;
    type Output = Array;
    type Error = Exception;
    fn begin_pass(&mut self, pass: usize) -> Result<(), Exception> {
        self.claim.begin_pass(pass)?;
        if pass == 1 {
            self.claim
                .accumulator()
                .as_mut()
                .expect("initialized scan")
                .begin_value_pass()?;
        }
        Ok(())
    }
    fn open_blocks(&mut self) -> Result<usize, Exception> {
        Ok(0)
    }
    fn next_block(&mut self, cursor: &mut usize) -> Result<Option<usize>, Exception> {
        let next = self.claim.acquire_next(self.stream)?;
        if let Some(index) = next {
            if index != *cursor {
                return Err(self.claim.proof.error(CacheSourceError::Identity));
            }
            *cursor += 1;
        }
        Ok(next)
    }
    fn consume_block(&mut self, index: &usize) -> Result<(), Exception> {
        let (start, end, bytes) = self.claim.accumulate_block(*index, self.stream)?;
        self.account(start, end, bytes);
        Ok(())
    }
    fn submit(&mut self) -> Result<(), Exception> {
        self.claim
            .accumulator()
            .as_ref()
            .expect("initialized scan")
            .submit()?;
        self.claim.release_completed_block()
    }
    fn consume_tail(&mut self) -> Result<(), Exception> {
        if let Some((start, end, bytes)) = self.claim.accumulate_tail(self.stream)? {
            self.account(start, end, bytes);
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<Array, Exception> {
        let output = self
            .claim
            .accumulator()
            .as_ref()
            .expect("one scan finish")
            .finish_retained(self.stream)?;
        let output = self.claim.retain_output(output)?;
        crate::backend::runtime::cache::complete_values([&output], self.stream)?;
        self.claim.proof.manager().record_original_attention_scan(
            &self.claim.proof,
            self.queries.dim(-2) > 1,
            self.scanned_blocks,
            self.scanned_bytes,
            self.scratch,
        )?;
        // The actual recurrence and its input aliases remain in Q until the
        // final output completed and its exact manager report succeeded.
        self.claim.accumulator().take();
        self.claim.discard_completed_output(&output)?;
        Ok(output)
    }
}

pub(crate) fn control_bytes() -> Option<usize> {
    use std::mem::size_of;
    let frames = [
        PagedScanPlan::control_bytes::<OriginalScan<'_, '_>>()?,
        size_of::<OriginalScan<'_, '_>>(),
        size_of::<BlockwiseAttentionAccumulator>(),
        size_of::<Result<BlockwiseAttentionAccumulator, Exception>>(),
        size_of::<(
            &mut PagedKeyValueCache,
            &Array,
            f32,
            Option<&Array>,
            Option<&Array>,
            Option<f32>,
            AttentionArithmetic,
            Option<&crate::backend::nn::relative_attention::RelativeAttentionKernel>,
            &Stream,
        )>(),
        size_of::<(
            PagedScanInput<'_>,
            CacheResidencyManager,
            PagedScanPlan,
            BlockwiseAttentionOptions,
        )>(),
        size_of::<(&mut OriginalPagedScanClaim<'_>, &Array, &Stream)>(),
        size_of::<(i64, i64, u64)>(),
        size_of::<Result<Option<(i64, i64, u64)>, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<OriginalScopeObserver>(),
        size_of::<Option<usize>>(),
        // Same shared block-account helper arguments, with no vector allocation.
        size_of::<(&Array, i64, i64, u64, &mut u64, &mut u64, &mut u64)>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
