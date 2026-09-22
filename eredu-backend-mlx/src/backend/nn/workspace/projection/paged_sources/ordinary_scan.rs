//! Exact ordinary scan inventory from retained canonical and append identities.
use super::scan_program::ScanSourcePair;
use crate::backend::{
    nn::workspace::{OrdinaryCallControls, OrdinaryNativeControls},
    runtime::cache::{
        kv::{BlockwiseAttentionAccumulator, ProjectedPagedSource},
        residency::{
            CacheBlockMetadata, CacheSourceError, CacheSourceFailure, InstalledManagerCatalog,
            PinnedCacheBlock, PinnedCacheBlockLease, PreparedCacheTransferStream,
        },
    },
};
use eredu_core::cache::CacheBlockId;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
use eredu_runtime::{
    cache::{PagedAppendPlan, PagedScanPlan},
    CacheBlockSelection, CacheStoragePhase,
};
use safemlx::Array;
use std::mem::{size_of, size_of_val};

/// Descriptive destinations retain their actual manager stream owner. Native
/// completion, promotion and publication grants remain separate sources.
pub(crate) struct PreparedOrdinaryPagedScan {
    pub(crate) ids: Vec<CacheBlockId>,
    pub(crate) rows: Vec<OrdinaryScanRow>,
    pub(crate) tail: Option<ScanSourcePair>,
    pub(crate) accumulator: Option<BlockwiseAttentionAccumulator>,
    pub(crate) failed_root: Option<Array>,
    pub(crate) active_lease: Option<PinnedCacheBlockLease>,
    resident: bool,
    host_itinerary: bool,
    discard_ids: Vec<CacheBlockId>,
    discard_frontier: Option<(i64, i64)>,
    pub(crate) transfer: PreparedCacheTransferStream,
    pub(crate) query_start: i64,
    pub(crate) context_end: i64,
    pub(crate) tail_start: i64,
    _funding: Option<HostMetadataFunding>,
}
pub(crate) struct OrdinaryScanRow {
    pub(crate) pair: ScanSourcePair,
    pub(crate) pin: Option<PinnedCacheBlock>,
}
impl PreparedOrdinaryPagedScan {
    pub(super) fn prepare<'a, I>(
        source: &ProjectedPagedSource,
        append: &PagedAppendPlan,
        ids: I,
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure>
    where
        I: Iterator<Item = &'a CacheBlockId> + Clone,
    {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let frames = [
            size_of::<Self>(),
            size_of::<(&ProjectedPagedSource, &PagedAppendPlan, &WorkspaceContext)>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<I>()
                .checked_mul(3)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            size_of::<Option<(i64, i64)>>(),
            size_of::<Vec<CacheBlockId>>(),
            size_of::<&[CacheBlockId]>(),
            size_of::<PagedScanPlan>(),
            size_of::<(i64, i32, i64)>(),
            size_of::<CacheBlockId>(),
            size_of::<std::slice::Iter<'_, CacheBlockId>>(),
            size_of::<OrdinaryScanRow>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Result<ScanSourcePair, CacheSourceFailure>>(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let geometry = source.append_geometry();
        if geometry.block_size <= 0 {
            return Err(fail(CacheSourceError::Geometry));
        }
        let (tail_start, _, context_end) = append.resulting_tail();
        let query_start = append.initial_frontier().2;
        let queries = context_end
            .checked_sub(query_start)
            .and_then(|count| i32::try_from(count).ok())
            .ok_or_else(|| fail(CacheSourceError::Geometry))?;
        // Visibility is independent of the selected arithmetic pass count.
        // Numerical source qualification still comes from the actual trace.
        let selection = PagedScanPlan::new(
            context_end,
            queries,
            geometry.window,
            i64::from(geometry.prefix_tokens),
            eredu_nn::BlockwiseAttentionOptions::default(),
        )
        .map_err(|cause| fail(CacheSourceError::Scan(cause)))?;
        let count = ids
            .clone()
            .filter(|id| selection.selects(id.start, id.end))
            .try_fold(0usize, |count, _| count.checked_add(1))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let mut selected: Vec<CacheBlockId> = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        let retained = source.geometry();
        for id in ids.clone().filter(|id| selection.selects(id.start, id.end)) {
            if id.session_id != retained.session_id
                || id.global_layer != retained.global_layer
                || id.rank != retained.rank
                || id.start < 0
                || id.end <= id.start
                || selected.last().is_some_and(|prior| prior.end > id.start)
            {
                return Err(fail(CacheSourceError::Identity));
            }
            selected.push(id.clone());
        }
        let discard_frontier = geometry
            .window
            .filter(|_| {
                !source
                    .manager()
                    .options()
                    .retains_discarded_for_persistence()
            })
            .map(|window| {
                (
                    (context_end - i64::from(window)).max(i64::from(geometry.prefix_tokens)),
                    i64::from(geometry.prefix_tokens),
                )
            });
        let discarded = |id: &&CacheBlockId| {
            discard_frontier.is_some_and(|(start, prefix)| {
                CacheBlockSelection::outside_retained_window(id.start, id.end, start, prefix)
            })
        };
        let discard_count = ids
            .clone()
            .filter(discarded)
            .try_fold(0usize, |count, _| count.checked_add(1))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let mut discard_ids = context
            .metadata_vec(discard_count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for id in ids.filter(discarded) {
            discard_ids.push(id.clone());
        }
        let transfer = source
            .manager()
            .prepared_transfer_stream()
            .map_err(|cause| fail(CacheSourceError::TransferSource(cause)))?;
        let mut rows = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for _ in 0..count {
            rows.push(OrdinaryScanRow {
                pair: ScanSourcePair::prepare(context)?,
                pin: None,
            });
        }
        let tail = (context_end > tail_start)
            .then(|| ScanSourcePair::prepare(context))
            .transpose()?;
        // Full resident scans need no staging or promotion. Other strategies
        // retain this exact inventory but require their own transfer source.
        let resident = !geometry.key_only
            && retained
                .blocks
                .iter()
                .all(|block| block.phase == CacheStoragePhase::Device);
        Ok(Self {
            ids: selected,
            rows,
            tail,
            accumulator: None,
            failed_root: None,
            active_lease: None,
            resident,
            host_itinerary: !geometry.key_only && source.host_trace().is_some(),
            discard_ids,
            discard_frontier,
            transfer,
            query_start,
            context_end,
            tail_start,
            _funding: context.metadata_funding(),
        })
    }
    pub(crate) fn bind_sources(
        &mut self,
        source: &ProjectedPagedSource,
        installed: &InstalledManagerCatalog,
        tail: Option<[&Array; 2]>,
        use_itinerary: bool,
    ) -> Result<(), CacheSourceError> {
        if !(if use_itinerary {
            self.host_itinerary
        } else {
            self.resident
        }) || !source.manager().same_catalog(installed.manager())
            || source.geometry().generation != installed.initial_generation()
            || self.tail.is_some() != tail.is_some()
            || self.rows.len() != self.ids.len()
        {
            return Err(CacheSourceError::Identity);
        }
        let geometry = source.append_geometry();
        // A Host itinerary owns both hot and promoted row pins. Binding a
        // second pin set here would prevent its exact eviction proof.
        for (id, row) in
            self.ids
                .iter()
                .zip(&mut self.rows)
                .take(if use_itinerary { 0 } else { self.ids.len() })
        {
            if row.pin.is_some() || row.pair.values().is_some() {
                return Err(CacheSourceError::Identity);
            }
            installed.with_source(
                CacheBlockSelection::new(id.global_layer, id.representation, id.start, id.end, 0),
                |mut loan| {
                    if loan.blocks().count() != 1 || !loan.blocks().any(|block| block.id() == id) {
                        return Err(CacheSourceError::Identity);
                    }
                    // Pin before filling: even a partial clone failure keeps the
                    // backing and its physical charge in this retained row.
                    row.pin = Some(loan.pin_prepared_block(id, self._funding.clone())?);
                    let block = loan.blocks().next().ok_or(CacheSourceError::Identity)?;
                    let arrays = block.device().ok_or(CacheSourceError::PromotionRequired)?;
                    validate_arrays(arrays, geometry, id.end - id.start)?;
                    row.pair.fill_for_inspection(arrays, true)
                },
            )?;
        }
        if let (Some(pair), Some(arrays)) = (&mut self.tail, tail) {
            validate_arrays(arrays, geometry, self.context_end - self.tail_start)?;
            pair.fill_for_inspection(arrays, false)?;
        }
        Ok(())
    }

    /// Only called after final synchronous completion and the manager report.
    /// The enclosing Work retains every partially filled row on any error.
    fn retire_settled(&mut self) {
        self.accumulator = None;
        self.failed_root = None;
        for row in &mut self.rows {
            row.pair.arrays = [None, None];
        }
        if let Some(tail) = &mut self.tail {
            tail.arrays = [None, None];
        }
        self.active_lease = None;
        for row in &mut self.rows {
            row.pin = None;
        }
    }

    pub(crate) fn finish_completed(
        &mut self,
        source: &ProjectedPagedSource,
        installed: &InstalledManagerCatalog,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Array, safemlx::error::Exception> {
        let fail = |cause| super::OrdinaryPagedWork::source_error(cause, host);
        if self.active_lease.is_some()
            || !self
                .accumulator
                .as_ref()
                .is_some_and(BlockwiseAttentionAccumulator::has_completed_recurrence)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let output = self
            .failed_root
            .as_ref()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if output
            .try_descriptor()
            .map_err(|cause| fail(CacheSourceError::Descriptor(cause)))?
            .facts()
            .allocation()
            .is_none()
        {
            return Err(fail(CacheSourceError::PendingStorage));
        }
        let output = self.failed_root.take().expect("validated completed root");
        self.retire_settled();
        let claim = OrdinaryDiscard {
            source,
            installed,
            host,
            ids: &self.discard_ids,
            frontier: self.discard_frontier,
            funding: &self._funding,
        };
        source.manager().discard_prepared(&claim)?;
        Ok(output)
    }

    pub(super) fn caller_controls(&self) -> Option<OrdinaryCallControls> {
        if !self.resident && !self.host_itinerary {
            return None;
        }
        let pairs = self
            .rows
            .len()
            .checked_add(usize::from(self.tail.is_some()))?;
        let frames = [
            size_of::<(
                &mut Self,
                &ProjectedPagedSource,
                &InstalledManagerCatalog,
                Option<[&Array; 2]>,
                bool,
            )>(),
            size_of::<Result<(), CacheSourceError>>(),
            size_of::<
                std::iter::Zip<
                    std::slice::Iter<'_, CacheBlockId>,
                    std::slice::IterMut<'_, OrdinaryScanRow>,
                >,
            >(),
            ScanSourcePair::inspection_control_bytes()?.checked_mul(pairs)?,
            PinnedCacheBlock::fixed_controls()?.checked_mul(self.rows.len().checked_mul(3)?)?,
            InstalledManagerCatalog::source_control_bytes::<()>(size_of::<(
                &CacheBlockId,
                &mut OrdinaryScanRow,
                &Option<HostMetadataFunding>,
                eredu_runtime::working_memory::WorkspacePagedGeometry,
            )>())?
            .checked_mul(self.rows.len())?,
            InstalledManagerCatalog::report_control_bytes()?,
            crate::backend::runtime::cache::residency::CacheResidencyManager::prepared_discard_control_bytes::<OrdinaryDiscard<'_>>()?
                .checked_mul(self.discard_ids.len().checked_mul(2)?.checked_add(1)?)?,
            size_of::<OrdinaryDiscard<'_>>(),
            Array::descriptor_control_bytes()?,
            size_of::<(&mut Self, &ProjectedPagedSource, &InstalledManagerCatalog, &eredu_core::HostPreparationAuthority)>(),
            size_of::<Result<Array, safemlx::error::Exception>>(),
            size_of::<([&Array; 2], eredu_runtime::working_memory::WorkspacePagedGeometry, i64)>(),
            size_of::<[[i32; 4]; 2]>(),
            size_of::<[i32; 4]>(),
            // Both defined arithmetic policies use at most two visits per
            // actual source pair. This bounds caller aliases, not tensor bytes.
            Array::ordinary_clone_control_bytes()?.checked_mul(pairs.checked_mul(4)?)?,
            crate::backend::runtime::cache::kv::PagedKeyValueCache::ordinary_scan_control_bytes()?,
        ];
        Some(OrdinaryCallControls {
            metadata_bytes: u64::try_from(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)?,
            )
            .ok()?,
            observed: OrdinaryNativeControls::default(),
        })
    }
}
fn validate_arrays(
    arrays: [&Array; 2],
    geometry: eredu_runtime::working_memory::WorkspacePagedGeometry,
    tokens: i64,
) -> Result<(), CacheSourceError> {
    let [batch, heads, width] = geometry.dimensions;
    let tokens = i32::try_from(tokens).map_err(|_| CacheSourceError::Overflow)?;
    let shapes = [
        [batch, heads, tokens, width],
        [
            batch,
            heads,
            tokens,
            if geometry.key_only { 1 } else { width },
        ],
    ];
    if arrays[0].shape() != shapes[0]
        || arrays[1].shape() != shapes[1]
        || arrays[0].dtype() != arrays[1].dtype()
        || CacheBlockMetadata::floating_dtype_bytes(arrays[0].dtype()).is_none()
    {
        return Err(CacheSourceError::Geometry);
    }
    Ok(())
}

/// The sole constructor is the completed row's final output handoff above.
struct OrdinaryDiscard<'a> {
    source: &'a ProjectedPagedSource,
    installed: &'a InstalledManagerCatalog,
    host: &'a eredu_core::HostPreparationAuthority,
    ids: &'a [CacheBlockId],
    frontier: Option<(i64, i64)>,
    funding: &'a Option<HostMetadataFunding>,
}
impl crate::backend::runtime::cache::residency::PreparedCacheDiscard for OrdinaryDiscard<'_> {
    type Cause = super::OrdinaryPagedCause;
    fn error(&self, cause: impl Into<Self::Cause>) -> safemlx::error::Exception {
        super::OrdinaryPagedWork::source_error(cause, self.host)
    }
    fn validate_manager(
        &self,
        manager: &crate::backend::runtime::cache::residency::CacheResidencyManager,
        generation: u64,
    ) -> Result<(), safemlx::error::Exception> {
        if !manager.same_catalog(self.source.manager())
            || !manager.same_catalog(self.installed.manager())
            || generation != self.installed.initial_generation()
        {
            return Err(self.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    fn layer(&self) -> usize {
        self.source.geometry().global_layer
    }
    fn ids(&self) -> &[CacheBlockId] {
        self.ids
    }
    fn frontier(&self) -> Option<(i64, i64)> {
        self.frontier
    }
    fn funding(&self) -> Option<HostMetadataFunding> {
        self.funding.clone()
    }
}
