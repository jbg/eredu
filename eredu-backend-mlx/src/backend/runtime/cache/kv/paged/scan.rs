//! Actual manager leases and the existing native blockwise numerical worker.
use super::*;
mod ordinary;
mod original;
use crate::backend::runtime::cache::residency::{CacheBlockLease, CacheBlockPrefetch};
use eredu_runtime::cache::{PagedScanMechanisms, PagedScanPlan};
pub(super) use ordinary::{control_bytes as ordinary_control_bytes, run as ordinary_scan};
pub(super) use original::control_bytes as original_control_bytes;
pub(super) use original::run as original_scan;

pub(super) struct NativeScan<'a> {
    pub cache: &'a mut PagedKeyValueCache,
    pub queries: &'a Array,
    pub ids: Vec<CacheBlockId>,
    pub accumulator: Option<BlockwiseAttentionAccumulator>,
    pub relative: Option<&'a crate::backend::nn::relative_attention::RelativeAttentionKernel>,
    pub stream: &'a Stream,
    pub scanned_blocks: u64,
    pub scanned_bytes: u64,
    pub scratch: u64,
}
impl NativeScan<'_> {
    fn accumulate(&mut self, block: KeyValueAttentionBlock, bytes: u64) -> Result<(), Exception> {
        account_block(
            self.queries,
            block.start,
            block.end,
            bytes,
            &mut self.scanned_blocks,
            &mut self.scanned_bytes,
            &mut self.scratch,
        );
        let bias = self
            .relative
            .map(|kernel| kernel.bias(block.start, block.end, self.stream))
            .transpose()?;
        if let Some(bias) = &bias {
            self.scratch = self.scratch.max(
                self.relative
                    .expect("bias has its actual kernel")
                    .prepared_bytes()
                    + 3 * bias.nbytes() as u64,
            );
        }
        self.accumulator
            .as_mut()
            .expect("scan has not finished")
            .accumulate_with_bias(&block, bias.as_ref(), self.stream)
    }
}
impl PagedScanMechanisms for NativeScan<'_> {
    type Cursor = CacheBlockPrefetch;
    type Block = CacheBlockLease;
    type Output = Array;
    type Error = Exception;
    fn begin_pass(&mut self, pass: usize) -> Result<(), Exception> {
        if pass == 1 {
            self.accumulator
                .as_mut()
                .expect("scan has not finished")
                .begin_value_pass()?;
        }
        Ok(())
    }
    fn open_blocks(&mut self) -> Result<Self::Cursor, Exception> {
        self.cache
            .manager
            .prefetch_blocks(self.ids.clone(), self.stream)
            .map_err(cache_residency_exception)
    }
    fn next_block(&mut self, cursor: &mut Self::Cursor) -> Result<Option<Self::Block>, Exception> {
        cursor.next_block().map_err(cache_residency_exception)
    }
    fn consume_block(&mut self, lease: &Self::Block) -> Result<(), Exception> {
        let id = lease.id();
        let (keys, values) = match lease.arrays() {
            CacheBlockArrays::KeyValue { keys, values } => (keys.clone(), values.clone()),
            _ => {
                return Err(Exception::custom(
                    "paged key/value cache found an incompatible block representation",
                ));
            }
        };
        let block = KeyValueAttentionBlock::unleased(id.start, id.end, keys, values);
        self.accumulate(block, lease.bytes())
    }
    fn submit(&mut self) -> Result<(), Exception> {
        self.accumulator
            .as_ref()
            .expect("scan has not finished")
            .submit()
    }
    fn consume_tail(&mut self) -> Result<(), Exception> {
        if let (Some(keys), Some(values)) = (&self.cache.tail_keys, &self.cache.tail_values) {
            let block = KeyValueAttentionBlock::unleased(
                self.cache.tail_start,
                self.cache.offset,
                keys.clone(),
                values.clone(),
            );
            let bytes = block.bytes;
            self.accumulate(block, bytes)?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<Array, Exception> {
        let output = self
            .accumulator
            .take()
            .expect("one shared scan finish")
            .finish(self.stream)?;
        crate::backend::runtime::cache::residency::evaluate_cache_arrays([&output])?;
        // Earlier queries in this span must consume their visible keys before
        // the final query's window is allowed to discard any source block.
        self.cache.discard_sliding_history()?;
        self.cache
            .manager
            .record_attention_scan(
                self.cache.global_layer,
                self.queries.dim(-2) > 1,
                self.scanned_blocks,
                self.scanned_bytes,
                self.scratch,
            )
            .map_err(cache_residency_exception)?;
        Ok(output)
    }
}

pub(super) fn plan(
    cache: &PagedKeyValueCache,
    queries: &Array,
    options: eredu_nn::BlockwiseAttentionOptions,
) -> Result<PagedScanPlan, Exception> {
    PagedScanPlan::new(
        cache.offset,
        queries.dim(-2),
        cache.sliding_window,
        i64::from(cache.prefix_tokens),
        options,
    )
    .map_err(|cause| Exception::custom(cause.to_string()))
}

fn account_block(
    queries: &Array,
    start: i64,
    end: i64,
    bytes: u64,
    blocks: &mut u64,
    scanned: &mut u64,
    scratch: &mut u64,
) {
    *scratch = (*scratch).max(
        queries.dim(0) as u64
            * queries.dim(1) as u64
            * queries.dim(-2) as u64
            * (end - start) as u64
            * 4,
    );
    *blocks += 1;
    *scanned += bytes;
}
