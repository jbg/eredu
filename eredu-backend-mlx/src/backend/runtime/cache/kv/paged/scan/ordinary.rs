//! Ordinary source adapter for the shared paged traversal.
use super::*;
use crate::backend::nn::{
    shared::OrdinaryExecutionOwner,
    workspace::{
        OrdinaryPagedHostScan, OrdinaryPagedWork, PagedScanInput, PreparedOrdinaryPagedScan,
    },
};
use crate::backend::runtime::cache::residency::{CacheSourceError, InstalledManagerCatalog};
use eredu_core::HostPreparationAuthority;
use eredu_nn::{AttentionArithmetic, BlockwiseAttentionOptions};

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
    owner: OrdinaryExecutionOwner,
) -> Result<Option<Array>, Exception> {
    let host = owner.host();
    let fail = |cause| OrdinaryPagedWork::source_error(cause, host);
    if queries.ndim() != 4 {
        return Err(fail(CacheSourceError::Geometry));
    }
    if relative.is_some() {
        return Err(fail(CacheSourceError::RelativeScan));
    }
    let work = owner
        .paged()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
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
        block_size: manager.options().block_size_tokens(),
        offset: cache.offset,
        tail_start: cache.tail_start,
        window: cache.sliding_window,
        prefix: cache.prefix_tokens,
        options,
    };
    work.with_host_scan(actual, host, |source, installed, scan, host_scan| {
        let tail = match (&cache.tail_keys, &cache.tail_values) {
            (Some(keys), Some(values)) => Some([keys, values]),
            (None, None) => None,
            _ => return Err(fail(CacheSourceError::Geometry)),
        };
        let use_itinerary = host_scan.has_itinerary()?;
        scan.bind_sources(source, installed, tail, use_itinerary)
            .map_err(fail)?;
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
        scan.accumulator = Some(accumulator);
        plan.run(&mut OrdinaryScan {
            scan,
            source,
            installed,
            host,
            host_scan: use_itinerary.then_some(host_scan),
            pass: 0,
            queries,
            stream,
            layer: cache.global_layer,
            scanned_blocks: 0,
            scanned_bytes: 0,
            scratch: 0,
        })
    })
    .map(Some)
}

struct OrdinaryScan<'a> {
    scan: &'a mut PreparedOrdinaryPagedScan,
    source: &'a crate::backend::runtime::cache::kv::ProjectedPagedSource,
    installed: &'a InstalledManagerCatalog,
    host: &'a HostPreparationAuthority,
    host_scan: Option<OrdinaryPagedHostScan<'a>>,
    pass: usize,
    queries: &'a Array,
    stream: &'a Stream,
    layer: usize,
    scanned_blocks: u64,
    scanned_bytes: u64,
    scratch: u64,
}
impl OrdinaryScan<'_> {
    fn error(&self, cause: CacheSourceError) -> Exception {
        OrdinaryPagedWork::source_error(cause, self.host)
    }
    fn accumulate(&mut self, block: KeyValueAttentionBlock) -> Result<(), Exception> {
        let scratch = [
            self.queries.dim(0),
            self.queries.dim(1),
            self.queries.dim(-2),
        ]
        .into_iter()
        .try_fold(4u64, |n, dimension| {
            n.checked_mul(u64::try_from(dimension).ok()?)
        })
        .and_then(|n| n.checked_mul(u64::try_from(block.end.checked_sub(block.start)?).ok()?))
        .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        self.scanned_blocks = self
            .scanned_blocks
            .checked_add(1)
            .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        self.scanned_bytes = self
            .scanned_bytes
            .checked_add(block.bytes)
            .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        self.scratch = self.scratch.max(scratch);
        self.scan
            .accumulator
            .as_mut()
            .expect("initialized ordinary scan")
            .accumulate_with_bias(&block, None, self.stream)
    }
}
impl PagedScanMechanisms for OrdinaryScan<'_> {
    type Cursor = usize;
    type Block = usize;
    type Output = Array;
    type Error = Exception;
    fn begin_pass(&mut self, pass: usize) -> Result<(), Exception> {
        if self.scan.active_lease.is_some() {
            return Err(self.error(CacheSourceError::Identity));
        }
        if pass == 1 {
            self.scan
                .accumulator
                .as_mut()
                .expect("initialized ordinary scan")
                .begin_value_pass()?;
        }
        self.pass = pass;
        Ok(())
    }
    fn open_blocks(&mut self) -> Result<usize, Exception> {
        Ok(0)
    }
    fn next_block(&mut self, cursor: &mut usize) -> Result<Option<usize>, Exception> {
        if self.scan.active_lease.is_some() {
            return Err(self.error(CacheSourceError::Identity));
        }
        let Some(row) = self.scan.rows.get(*cursor) else {
            return Ok(None);
        };
        if self.host_scan.is_none() {
            let pin = row
                .pin
                .as_ref()
                .ok_or_else(|| self.error(CacheSourceError::Identity))?;
            self.scan.active_lease = Some(pin.acquire().map_err(|cause| self.error(cause))?);
        }
        let index = *cursor;
        *cursor = cursor
            .checked_add(1)
            .ok_or_else(|| self.error(CacheSourceError::Overflow))?;
        Ok(Some(index))
    }
    fn consume_block(&mut self, index: &usize) -> Result<(), Exception> {
        if let Some(loan) = self.host_scan.take() {
            let id = self.scan.ids[*index].clone();
            let result =
                loan.with_acquired(&id, self.pass, self.stream, |[keys, values], lease| {
                    self.scan.active_lease = Some(lease);
                    let block = KeyValueAttentionBlock::unleased(
                        id.start,
                        id.end,
                        keys.try_clone_handle()?,
                        values.try_clone_handle()?,
                    );
                    self.accumulate(block)
                });
            self.host_scan = Some(loan);
            return result;
        }
        if self.scan.active_lease.is_none() {
            return Err(self.error(CacheSourceError::Identity));
        }
        let id = &self.scan.ids[*index];
        let [keys, values] = self.scan.rows[*index]
            .pair
            .values()
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let block = KeyValueAttentionBlock::unleased(
            id.start,
            id.end,
            keys.try_clone_handle()?,
            values.try_clone_handle()?,
        );
        self.accumulate(block)
    }
    fn submit(&mut self) -> Result<(), Exception> {
        let accumulator = self
            .scan
            .accumulator
            .as_ref()
            .expect("initialized ordinary scan");
        if !accumulator.has_completed_recurrence() {
            return Err(self.error(CacheSourceError::PendingStorage));
        }
        accumulator.submit()?;
        self.scan.active_lease = None;
        Ok(())
    }
    fn consume_tail(&mut self) -> Result<(), Exception> {
        if let Some(pair) = &self.scan.tail {
            let [keys, values] = pair
                .values()
                .ok_or_else(|| self.error(CacheSourceError::Identity))?;
            let block = KeyValueAttentionBlock::unleased(
                self.scan.tail_start,
                self.scan.context_end,
                keys.try_clone_handle()?,
                values.try_clone_handle()?,
            );
            self.accumulate(block)?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<Array, Exception> {
        let output = self
            .scan
            .accumulator
            .as_ref()
            .expect("one ordinary scan finish")
            .finish_retained(self.stream)?;
        self.scan.failed_root = Some(output);
        let output = self.scan.failed_root.as_ref().expect("retained final root");
        crate::backend::runtime::cache::residency::evaluate_cache_arrays([output])?;
        self.installed
            .record_scan(
                self.layer,
                self.queries.dim(-2) > 1,
                self.scanned_blocks,
                self.scanned_bytes,
                self.scratch,
            )
            .map_err(|cause| self.error(cause))?;
        self.scan
            .finish_completed(self.source, self.installed, self.host)
    }
}

pub(crate) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        PagedScanPlan::control_bytes::<OrdinaryScan<'_>>()?,
        size_of::<OrdinaryScan<'_>>(),
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
            OrdinaryExecutionOwner,
        )>(),
        size_of::<(
            PagedScanInput<'_>,
            CacheResidencyManager,
            PagedScanPlan,
            BlockwiseAttentionOptions,
        )>(),
        size_of::<(
            &mut PreparedOrdinaryPagedScan,
            &InstalledManagerCatalog,
            &Array,
            &Stream,
            &HostPreparationAuthority,
            &mut PagedKeyValueCache,
            f32,
            Option<&Array>,
            Option<&Array>,
        )>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Option<OrdinaryExecutionOwner>>(),
        size_of::<Result<Option<OrdinaryExecutionOwner>, Exception>>(),
        size_of::<KeyValueAttentionBlock>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<(usize, i64, i64, u64)>(),
        size_of::<CacheBlockId>(),
        OrdinaryPagedHostScan::source_control_bytes::<()>(size_of::<(
            &mut OrdinaryScan<'_>,
            &CacheBlockId,
        )>())?,
        size_of::<std::array::IntoIter<i32, 3>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
