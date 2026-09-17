//! Exact old-window source and fixed destinations before the accepted append.
use super::programs::PagedAppendProgram;
use super::scan_program::{ScanSourcePair, ScanSourceRow};
use super::*;
use crate::backend::runtime::cache::{
    kv::ProjectedPagedSource,
    residency::{CacheSourceError, CacheSourceFailure, PinnedCacheBlock, PinnedCacheBlockLease},
};
use eredu_core::cache::CacheBlockId;
use eredu_runtime::cache::{PagedLocalUpdatePlan, PagedVisiblePlan};
use safemlx::Array;
use std::mem::size_of;

pub(super) struct PreparedPagedVisible {
    // Derived native arrays retire before their demand/source leases; the
    // source account follows every backing and failed-prefix witness.
    pub(super) keys: Vec<Array>,
    pub(super) values: Vec<Array>,
    pub(super) failed_root: Option<Array>,
    pub(super) tail: Option<ScanSourcePair>,
    pub(super) active_lease: Option<PinnedCacheBlockLease>,
    pub(super) rows: Vec<ScanSourceRow>,
    pub(super) plan: Option<PagedVisiblePlan>,
    pub(super) tail_start: i64,
    pub(super) offset: i64,
    pub(super) result_tokens: i32,
    pub(super) parts_limit: usize,
    pub(super) started: bool,
    pub(super) read_finished: bool,
    pub(super) completed: bool,
    _funding: Option<WorkspaceMetadataFunding>,
}

impl PreparedPagedVisible {
    pub(super) fn retire_settled(&mut self) {
        self.failed_root = None;
        self.keys.clear();
        self.values.clear();
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
}
pub(super) fn prepare_all(
    programs: &mut [Option<PagedAppendProgram>],
    sources: &[ProjectedPagedSource],
    context: &WorkspaceContext,
) -> Result<(), CacheSourceFailure> {
    context
        .charge_metadata(size_of::<(
            &mut [Option<PagedAppendProgram>],
            &[ProjectedPagedSource],
            &WorkspaceContext,
            std::ops::Range<usize>,
            Option<PreparedPagedVisible>,
            Result<Option<PreparedPagedVisible>, CacheSourceFailure>,
        )>())
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    for index in 0..programs.len() {
        let prepared = prepare_one(index, programs, sources, context)?;
        programs[index]
            .as_mut()
            .expect("retained append source")
            .visible = prepared;
    }
    Ok(())
}
fn prepare_one(
    index: usize,
    programs: &[Option<PagedAppendProgram>],
    sources: &[ProjectedPagedSource],
    context: &WorkspaceContext,
) -> Result<Option<PreparedPagedVisible>, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    let current = programs
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let source = sources
        .get(current.source)
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let geometry = source.geometry();
    let Some(window) = geometry.sliding_window else {
        return Ok(None);
    };
    let (tail_start, tail_len, offset) = current.plan.initial_frontier();
    let end = current.plan.resulting_tail().2;
    let inputs = end
        .checked_sub(offset)
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| fail(CacheSourceError::Geometry))?;
    let local = PagedLocalUpdatePlan::new(offset, window, inputs)
        .map_err(|cause| fail(CacheSourceError::Visible(cause)))?;
    let plan = local.history();
    let selected = |id: &CacheBlockId| plan.is_some_and(|plan| plan.selects(id.start, id.end));
    let ids = || {
        geometry.blocks.iter().map(|block| &block.id).chain(
            programs
                .iter()
                .flatten()
                .filter(move |program| {
                    program.source == current.source && program.ordinal < current.ordinal
                })
                .flat_map(|program| program.publications.iter().map(|row| &row.id)),
        )
    };
    let frames = [
        size_of::<PreparedPagedVisible>(),
        size_of::<(
            usize,
            &[Option<PagedAppendProgram>],
            &[ProjectedPagedSource],
            &WorkspaceContext,
        )>(),
        size_of::<(i64, i32, i64, i64, i32, usize, usize)>(),
        size_of::<Result<Option<PreparedPagedVisible>, CacheSourceFailure>>(),
        size_of::<PagedLocalUpdatePlan>(),
        size_of::<Option<PagedVisiblePlan>>(),
        size_of::<Result<PagedLocalUpdatePlan, eredu_runtime::cache::PagedVisibleError>>(),
        std::mem::size_of_val(&ids),
        std::mem::size_of_val(&ids()),
        std::mem::size_of_val(&selected),
        size_of::<std::slice::IterMut<'_, ScanSourceRow>>(),
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
    let count = ids().filter(|id| selected(id)).count();
    let has_tail = tail_len > 0 && plan.is_some_and(|plan| plan.selects(tail_start, offset));
    let parts_limit = count
        .checked_add(usize::from(has_tail))
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    context
        .charge_metadata(
            super::append_claim::visible_control_bytes(count, parts_limit, plan.is_some())
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        )
        .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
    let mut rows = context
        .metadata_vec(count)
        .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
    for id in ids().filter(|id| selected(id)) {
        if id.end > tail_start
            || id.global_layer != geometry.global_layer
            || id.session_id != geometry.session_id
            || id.rank != geometry.rank
            || rows
                .last()
                .is_some_and(|previous: &ScanSourceRow| previous.id.end > id.start)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        rows.push(ScanSourceRow {
            id: id.clone(),
            pair: ScanSourcePair::prepare(context)?,
            demotion: None,
            pin: None,
        });
    }
    let history = plan.map_or(0, |plan| plan.range().end - plan.range().start);
    let result_tokens = history
        .checked_add(i64::from(inputs))
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    Ok(Some(PreparedPagedVisible {
        plan,
        rows,
        tail: if has_tail {
            Some(ScanSourcePair::prepare(context)?)
        } else {
            None
        },
        tail_start,
        offset,
        result_tokens,
        parts_limit,
        keys: context
            .metadata_vec(parts_limit)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
        values: context
            .metadata_vec(parts_limit)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
        active_lease: None,
        failed_root: None,
        started: false,
        read_finished: plan.is_none(),
        completed: false,
        _funding: context.metadata_funding(),
    }))
}

pub(super) fn retirement_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<&mut PreparedPagedVisible>(),
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
