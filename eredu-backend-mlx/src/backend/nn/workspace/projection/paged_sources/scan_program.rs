//! One source-derived scan inventory; slots alone never grant native work.
use super::programs::PagedAppendProgram;
use super::*;
use crate::backend::runtime::cache::{
    kv::ProjectedPagedSource,
    residency::{CacheSourceError, CacheSourceFailure, PinnedCacheBlock, PinnedCacheBlockLease},
};
use eredu_core::cache::CacheBlockId;
use safemlx::{Array, PreparedArrayClone};
use std::mem::size_of;

pub(super) struct PreparedPagedScan {
    pub(super) accumulator:
        Option<crate::backend::runtime::cache::kv::BlockwiseAttentionAccumulator>,
    pub(super) failed_root: Option<Array>,
    pub(super) rows: Vec<ScanSourceRow>,
    pub(super) discard_ids: Vec<CacheBlockId>,
    pub(super) tail: Option<ScanSourcePair>,
    pub(super) active_lease: Option<PinnedCacheBlockLease>,
    pub(super) query_start: i64,
    pub(super) context_end: i64,
    pub(super) tail_start: i64,
    pub(super) used: bool,
    pub(super) completed: bool,
    _funding: Option<WorkspaceMetadataFunding>,
}
pub(super) struct ScanSourceRow {
    pub(super) id: CacheBlockId,
    pub(super) pair: ScanSourcePair,
    pub(super) demotion:
        Option<crate::backend::runtime::cache::residency::PreparedCacheHostDemotion>,
    pub(super) pin: Option<PinnedCacheBlock>,
}
/// These exact unfilled handles become owned source aliases only after an
/// authenticated native manager/local-tail loan fills them once.
pub(crate) struct ScanSourcePair {
    pub(super) arrays: [Option<Array>; 2],
    pub(super) slots: [PreparedArrayClone; 2],
    _funding: Option<WorkspaceMetadataFunding>,
}
pub(super) fn retirement_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<&mut PreparedPagedScan>(),
        size_of::<std::slice::IterMut<'_, ScanSourceRow>>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<Option<PinnedCacheBlock>>(),
        size_of::<Option<PinnedCacheBlockLease>>(),
        size_of::<Option<ScanSourcePair>>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}
impl PreparedPagedScan {
    pub(super) fn retire_settled(&mut self) {
        // No manager loan is held. All native aliases retire before canonical
        // source/demand pins; metadata and cumulative attempt flags remain.
        self.accumulator = None;
        self.failed_root = None;
        for row in &mut self.rows {
            row.pair.arrays = [None, None];
        }
        if let Some(tail) = &mut self.tail {
            tail.arrays = [None, None];
        }
        // Demotion rollback owners retain their exact replaced Device charge
        // through source-bank teardown. A later owner-attachment join can retire
        // that charge with the backing itself; settlement alone is no refund.
        self.active_lease = None;
        for row in &mut self.rows {
            row.pin = None;
        }
    }
}
impl ScanSourcePair {
    /// The caller keeps the actual source under its authenticated lexical loan.
    /// Every successful handle is installed before a later fill can fail.
    pub(crate) fn fill(
        &mut self,
        arrays: [&Array; 2],
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<(), safemlx::error::Exception> {
        if self.arrays.iter().any(Option::is_some) {
            return Err(observer.invalid_input_error());
        }
        for (index, array) in arrays.into_iter().enumerate() {
            self.arrays[index] = Some(self.slots[index].fill_in_original_scope(array, observer)?);
        }
        Ok(())
    }
    pub(crate) fn values(&self) -> Option<[&Array; 2]> {
        Some([self.arrays[0].as_ref()?, self.arrays[1].as_ref()?])
    }
    pub(super) fn prepare(context: &WorkspaceContext) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let one = Array::inspection_clone_handle_bytes()
            .checked_add(
                PreparedArrayClone::control_bytes()
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let frames = [
            size_of::<Self>(),
            size_of::<(&WorkspaceContext, usize)>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<[PreparedArrayClone; 2]>(),
            size_of::<Result<PreparedArrayClone, safemlx::PreparedArrayCloneCause>>(),
        ];
        let controls = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(one.checked_mul(2)?))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        context
            .charge_metadata(controls)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let slots = [
            PreparedArrayClone::try_prepare_for_inspection()
                .map_err(|cause| fail(CacheSourceError::Clone(cause)))?,
            PreparedArrayClone::try_prepare_for_inspection()
                .map_err(|cause| fail(CacheSourceError::Clone(cause)))?,
        ];
        Ok(Self {
            arrays: [None, None],
            slots,
            _funding: context.metadata_funding(),
        })
    }
}

pub(super) fn prepare_all(
    programs: &mut [Option<PagedAppendProgram>],
    sources: &[ProjectedPagedSource],
    context: &WorkspaceContext,
) -> Result<(), CacheSourceFailure> {
    context
        .charge_metadata(
            size_of::<(
                &mut [Option<PagedAppendProgram>],
                &[ProjectedPagedSource],
                &WorkspaceContext,
            )>()
            .checked_add(size_of::<std::ops::Range<usize>>())
            .and_then(|n| n.checked_add(size_of::<Result<PreparedPagedScan, CacheSourceFailure>>()))
            .ok_or_else(|| CacheSourceFailure::source(CacheSourceError::Overflow, context))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    for index in 0..programs.len() {
        let prepared = prepare_one(index, programs, sources, context)?;
        programs[index]
            .as_mut()
            .expect("prepared append program")
            .scan = Some(prepared);
    }
    Ok(())
}
fn prepare_one(
    index: usize,
    programs: &[Option<PagedAppendProgram>],
    sources: &[ProjectedPagedSource],
    context: &WorkspaceContext,
) -> Result<PreparedPagedScan, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let current = programs
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let source = sources
        .get(current.source)
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let geometry = source.geometry();
    let frames = [
        size_of::<PreparedPagedScan>(),
        size_of::<ScanSourceRow>(),
        size_of::<(
            usize,
            &[Option<PagedAppendProgram>],
            &[ProjectedPagedSource],
            &WorkspaceContext,
        )>(),
        size_of::<(&PagedAppendProgram, &ProjectedPagedSource)>(),
        size_of::<(usize, i64, i32, i64)>(),
        size_of::<Vec<ScanSourceRow>>(),
        size_of::<std::slice::Iter<'_, Option<PagedAppendProgram>>>(),
        size_of::<CacheBlockId>(),
        size_of::<Result<PreparedPagedScan, CacheSourceFailure>>(),
        PinnedCacheBlock::fixed_controls().ok_or_else(|| fail(CacheSourceError::Overflow))?,
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let (tail_start, tail_len, context_end) = current.plan.resulting_tail();
    let query_start = current.plan.initial_frontier().2;
    let queries = context_end.checked_sub(query_start)
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| fail(CacheSourceError::Geometry))?;
    let selection = eredu_runtime::cache::PagedScanPlan::new(
        context_end, queries, geometry.sliding_window,
        i64::from(geometry.prefix_tokens), eredu_nn::BlockwiseAttentionOptions::default(),
    ).map_err(|cause| fail(CacheSourceError::Scan(cause)))?;
    let discard_window = geometry.sliding_window.filter(|_| {
        !source.manager().options().retains_discarded_for_persistence()
    }).map(|window| {
        let prefix = i64::from(geometry.prefix_tokens);
        ((context_end - i64::from(window)).max(prefix), prefix)
    });
    let ids = || geometry.blocks.iter().map(|block| &block.id).chain(
        programs.iter().flatten()
            .filter(move |program| program.source == current.source && program.ordinal <= current.ordinal)
            .flat_map(|program| program.publications.iter().map(|entry| &entry.id))
    );
    let discarded = |id: &CacheBlockId| discard_window.is_some_and(|(visible, prefix)| {
        eredu_runtime::cache::CacheBlockSelection::outside_retained_window(
            id.start, id.end, visible, prefix,
        )
    });
    let selection_frames = [std::mem::size_of_val(&ids), std::mem::size_of_val(&ids()),
         std::mem::size_of_val(&discarded), std::mem::size_of_val(&selection),
         std::mem::size_of_val(&discard_window), size_of::<Vec<CacheBlockId>>(),
         size_of::<Result<eredu_runtime::cache::PagedScanPlan, eredu_runtime::cache::PagedScanError>>(),
         size_of::<(usize, usize, i32, i64)>(),
        ];
    context.charge_metadata(selection_frames.into_iter()
        .try_fold(std::mem::size_of_val(&selection_frames), usize::checked_add)
        .ok_or_else(|| fail(CacheSourceError::Overflow))?
    ).map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let count = ids().filter(|id| selection.selects(id.start, id.end)).count();
    let discard_count = ids().filter(|id| discarded(id)).count();
    // Both actual immediate calls and the later final-pin destructor are paid
    // before these exact identity records or deferred markers can be installed.
    let discard_controls = crate::backend::runtime::cache::residency::CacheResidencyManager::original_discard_control_bytes()
        .and_then(|n| n.checked_add(current.reporting_controls.checked_mul(2)?))
        .and_then(|n| n.checked_mul(discard_count.checked_mul(2)?.checked_add(1)?))
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    context.charge_metadata(discard_controls)
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut discard_ids = context.metadata_vec(discard_count)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    for id in ids().filter(|id| discarded(id)) {
        discard_ids.push(id.clone());
    }
    // One no-hooks manager binding per actual row, one final report, and one
    // active demand-lease slot reused across the policy's actual passes.
    let manager_calls = count
        .checked_add(1)
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    let controls = crate::backend::runtime::cache::residency::CacheResidencyManager::original_scan_control_bytes()
        .and_then(|n| n.checked_mul(manager_calls))
        .and_then(|n| n.checked_add(super::scan_claim::control_bytes()?))
        .and_then(|n| n.checked_add(crate::backend::runtime::cache::kv::PagedKeyValueCache::original_scan_control_bytes()?))
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    context
        .charge_metadata(controls)
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut rows = context
        .metadata_vec(count)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    let mut push = |id: &CacheBlockId| -> Result<(), CacheSourceFailure> {
        if rows
            .last()
            .is_some_and(|previous: &ScanSourceRow| previous.id.end > id.start)
            || id.global_layer != geometry.global_layer
            || id.session_id != geometry.session_id
            || id.rank != geometry.rank
            || id.start < 0
            || id.end <= id.start
        {
            return Err(fail(CacheSourceError::Identity));
        }
        rows.push(ScanSourceRow {
            id: id.clone(),
            pair: ScanSourcePair::prepare(context)?,
            demotion: None,
            pin: None,
        });
        Ok(())
    };
    context
        .charge_metadata(
            std::mem::size_of_val(&push)
                .checked_add(std::mem::size_of_val(&geometry.blocks.iter()))
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    for id in ids().filter(|id| selection.selects(id.start, id.end)) {
        push(id)?;
    }
    drop(push);
    Ok(PreparedPagedScan {
        accumulator: None,
        failed_root: None,
        rows,
        discard_ids,
        tail: if tail_len == 0 {
            None
        } else {
            Some(ScanSourcePair::prepare(context)?)
        },
        active_lease: None,
        query_start: current.plan.initial_frontier().2,
        context_end,
        tail_start,
        used: false,
        completed: false,
        _funding: context.metadata_funding(),
    })
}
