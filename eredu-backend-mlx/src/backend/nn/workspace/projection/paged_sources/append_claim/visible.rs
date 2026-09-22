//! Borrow the old-window source from the same one-use accepted append claim.
use super::super::{
    scan_claim::{OriginalPagedBlockSource, OriginalPagedDiscard, OriginalPagedScanSource},
    visible_program::PreparedPagedVisible,
};
use super::*;
use eredu_runtime::cache::{PagedVisibleError, PagedVisiblePlan};
use safemlx::{ops::indexing::TryIndexOp, Stream};

pub(crate) struct OriginalPagedVisibleClaim<'a, 'source> {
    proof: OriginalPagedScanSource<'source>,
    target: &'a mut PreparedPagedVisible,
    host: Option<&'a mut super::super::host_program::PreparedPagedHostProgram>,
    bank: &'source PagedSourceBank,
    ordinal: usize,
    roots: TransientRootsProjection,
    next: usize,
    host_load: Option<usize>,
}
impl OriginalPagedAppendClaim<'_> {
    pub(crate) fn with_visible<R>(
        &mut self,
        plan: PagedVisiblePlan,
        tail: Option<[&Array; 2]>,
        run: impl FnOnce(&mut OriginalPagedVisibleClaim<'_, '_>) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let proof = OriginalPagedScanSource {
            source: self.source,
            installed: self.installed,
            observer: self.observer.clone(),
            context: self.context,
            custody: self.custody.clone(),
        };
        let target = self
            .program
            .visible
            .as_mut()
            .ok_or_else(|| proof.error(CacheSourceError::Identity))?;
        if target.started
            || target.completed
            || target.plan.map(PagedVisiblePlan::range) != Some(plan.range())
        {
            return Err(proof.error(CacheSourceError::Identity));
        }
        target.started = true;
        if let Some(slot) = &mut target.tail {
            let arrays = tail.ok_or_else(|| proof.error(CacheSourceError::MissingHistory))?;
            proof.validate_arrays(arrays, target.offset - target.tail_start)?;
            for array in arrays {
                proof.observer.validate_completed_array(array)?;
            }
            slot.fill(arrays, &proof.observer)?;
        }
        let mut claim = OriginalPagedVisibleClaim {
            proof,
            target,
            host: self.host.as_deref_mut(),
            bank: self.bank,
            ordinal: self.program.ordinal,
            roots: self.roots.clone(),
            next: 0,
            host_load: None,
        };
        run(&mut claim)
    }
    pub(crate) fn retain_visible(&mut self, value: Array) -> Result<Array, Exception> {
        if let Err(cause) = self.roots.append(&value) {
            self.program.failed_root = Some(value);
            return Err(self.error(cause));
        }
        Ok(value)
    }
    pub(crate) fn complete_visible(
        &mut self,
        pair: [&Array; 2],
        stream: &Stream,
    ) -> Result<(), Exception> {
        let target = self
            .program
            .visible
            .as_ref()
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let geometry = self.source.append_geometry();
        let expected = [
            geometry.dimensions[0],
            geometry.dimensions[1],
            target.result_tokens,
            geometry.dimensions[2],
        ];
        if !target.read_finished
            || target.completed
            || target.active_lease.is_some()
            || pair[0].shape() != expected
            || pair[1].shape()
                != [
                    expected[0],
                    expected[1],
                    expected[2],
                    if geometry.key_only { 1 } else { expected[3] },
                ]
        {
            return Err(self.error(CacheSourceError::Geometry));
        }
        crate::backend::runtime::cache::complete_values(pair, stream)?;
        for array in pair {
            self.observer.validate_completed_array(array)?;
        }
        let target = self
            .program
            .visible
            .as_mut()
            .expect("checked visible source");
        target.started = true;
        target.completed = true;
        target.retire_settled();
        Ok(())
    }
    pub(crate) fn discard_after_visible(&self) -> Result<(), Exception> {
        let geometry = self.source.append_geometry();
        let Some(window) = geometry.window else {
            return Ok(());
        };
        self.validate_complete()?;
        if !self.visible_completed() {
            return Err(self.error(CacheSourceError::Identity));
        }
        let scan = self
            .program
            .scan
            .as_ref()
            .ok_or_else(|| self.error(CacheSourceError::Identity))?;
        let proof = OriginalPagedScanSource {
            source: self.source,
            installed: self.installed,
            observer: self.observer.clone(),
            context: self.context,
            custody: self.custody.clone(),
        };
        let prefix = i64::from(geometry.prefix_tokens);
        let end = self.program.plan.resulting_tail().2;
        let discard = OriginalPagedDiscard {
            proof: &proof,
            ids: &scan.discard_ids,
            frontier: Some(((end - i64::from(window)).max(prefix), prefix)),
        };
        proof.manager().discard_prepared(&discard)
    }
    pub(crate) fn visible_completed(&self) -> bool {
        self.program
            .visible
            .as_ref()
            .is_some_and(|target| target.completed)
    }
}
impl OriginalPagedVisibleClaim<'_, '_> {
    pub(crate) fn error(&self, cause: PagedVisibleError) -> Exception {
        self.proof.error(CacheSourceError::Visible(cause))
    }
    pub(crate) fn validate_plan(&self, plan: PagedVisiblePlan) -> Result<(), Exception> {
        if self.target.plan.map(PagedVisiblePlan::range) != Some(plan.range()) {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn acquire_next(&mut self, stream: &Stream) -> Result<Option<usize>, Exception> {
        if self.target.active_lease.is_some() || self.target.read_finished {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let Some(row) = self.target.rows.get_mut(self.next) else {
            return Ok(None);
        };
        if let Some(host) = self.host.as_deref_mut() {
            let (load, lease) = host.acquire(
                self.bank,
                &self.proof,
                self.ordinal,
                0,
                &row.id,
                &self.roots,
                stream,
            )?;
            self.host_load = Some(load);
            self.target.active_lease = Some(lease);
        } else {
            if row.pin.is_none() {
                if self.proof.source.has_host_promotion(&row.id) {
                    self.proof
                        .source
                        .promote_host(&row.id, &self.proof, &self.roots, stream)?;
                }
                let mut source = OriginalPagedBlockSource::new(
                    &self.proof,
                    &row.id,
                    &mut row.pair,
                    &mut row.pin,
                );
                self.proof.manager().bind_original_scan_block(&mut source)?;
            }
            self.target.active_lease = Some(
                row.pin
                    .as_ref()
                    .expect("bound row source")
                    .acquire()
                    .map_err(|cause| self.proof.error(cause))?,
            );
        }
        let index = self.next;
        self.next += 1;
        Ok(Some(index))
    }
    pub(crate) fn block_range(&self, index: usize) -> (i64, i64) {
        let id = &self.target.rows[index].id;
        (id.start, id.end)
    }
    fn arrays(&self, index: usize) -> Result<[&Array; 2], Exception> {
        if index.checked_add(1) != Some(self.next) || self.target.active_lease.is_none() {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        match self.host.as_deref() {
            Some(host) => host.values(
                self.host_load
                    .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?,
            ),
            None => self.target.rows[index].pair.values(),
        }
        .ok_or_else(|| self.proof.error(CacheSourceError::Identity))
    }
    fn retain(&mut self, value: Array) -> Result<Array, Exception> {
        if let Err(cause) = self.roots.append(&value) {
            self.target.failed_root = Some(value);
            return Err(self.proof.error(cause));
        }
        Ok(value)
    }
    fn check_part_slot(&self) -> Result<(), Exception> {
        if self.target.keys.len() != self.target.values.len()
            || self.target.keys.len() >= self.target.parts_limit
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        Ok(())
    }
    pub(crate) fn push_block(
        &mut self,
        index: usize,
        range: std::ops::Range<i32>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.check_part_slot()?;
        let keys = self.arrays(index)?[0].try_index_device((.., .., range.clone(), ..), stream)?;
        let keys = self.retain(keys)?;
        self.target.keys.push(keys);
        let values = self.arrays(index)?[1].try_index_device((.., .., range, ..), stream)?;
        let values = self.retain(values)?;
        self.target.values.push(values);
        Ok(())
    }
    pub(crate) fn submit_block(&mut self, stream: &Stream) -> Result<(), Exception> {
        if self.target.active_lease.is_none() || self.target.keys.len() != self.target.values.len()
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let keys = self
            .target
            .keys
            .last()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        let values = self
            .target
            .values
            .last()
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        crate::backend::runtime::cache::complete_values([keys, values], stream)?;
        self.target.active_lease = None;
        Ok(())
    }
    pub(crate) fn tail_range(&self) -> Option<(i64, i64)> {
        self.target
            .tail
            .as_ref()
            .map(|_| (self.target.tail_start, self.target.offset))
    }
    pub(crate) fn push_tail(
        &mut self,
        range: std::ops::Range<i32>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.check_part_slot()?;
        let arrays = self
            .target
            .tail
            .as_ref()
            .and_then(|pair| pair.values())
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        let keys = arrays[0].try_index_device((.., .., range.clone(), ..), stream)?;
        let keys = self.retain(keys)?;
        self.target.keys.push(keys);
        let arrays = self
            .target
            .tail
            .as_ref()
            .and_then(|pair| pair.values())
            .ok_or_else(|| self.proof.error(CacheSourceError::Identity))?;
        let values = arrays[1].try_index_device((.., .., range, ..), stream)?;
        let values = self.retain(values)?;
        self.target.values.push(values);
        Ok(())
    }
    pub(crate) fn finish(
        &mut self,
        tokens: i64,
        stream: &Stream,
    ) -> Result<(Array, Array), Exception> {
        if self.next != self.target.rows.len()
            || self.target.active_lease.is_some()
            || self.target.keys.len() != self.target.parts_limit
            || self.target.values.len() != self.target.parts_limit
            || self.target.parts_limit == 0
            || self.target.read_finished
        {
            return Err(self.proof.error(CacheSourceError::Identity));
        }
        let keys = if self.target.keys.len() == 1 {
            self.target.keys[0].try_clone_handle()?
        } else {
            safemlx::ops::concatenate_axis(self.target.keys.as_slice(), -2, stream)?
        };
        let keys = self.retain(keys)?;
        let values = if self.target.values.len() == 1 {
            self.target.values[0].try_clone_handle()?
        } else {
            safemlx::ops::concatenate_axis(self.target.values.as_slice(), -2, stream)?
        };
        let values = self.retain(values)?;
        if i64::from(keys.dim(-2)) != tokens || i64::from(values.dim(-2)) != tokens {
            return Err(self.proof.error(CacheSourceError::Geometry));
        }
        self.target.read_finished = true;
        Ok((keys, values))
    }
}

/// Caller and repeated leaf frames of this exact finite old-window inventory.
/// Native output allocation remains owned by the shared numerical source trace.
pub(in super::super) fn control_bytes(rows: usize, parts: usize, history: bool) -> Option<usize> {
    use std::mem::size_of;
    let fixed = [
        size_of::<OriginalPagedVisibleClaim<'_, '_>>(),
        size_of::<OriginalPagedScanSource<'_>>(),
        size_of::<OriginalPagedDiscard<'_, '_>>(),
        size_of::<(
            &mut OriginalPagedAppendClaim<'_>,
            PagedVisiblePlan,
            Option<[&Array; 2]>,
        )>(),
        size_of::<(&mut OriginalPagedAppendClaim<'_>, [&Array; 2], &Stream)>(),
        size_of::<(&OriginalPagedAppendClaim<'_>, i64, i64)>(),
        size_of::<(&mut OriginalPagedVisibleClaim<'_, '_>, i64, &Stream)>(),
        size_of::<([i32; 4], [i32; 4], WorkspacePagedGeometry)>(),
        size_of::<Result<(Array, Array), Exception>>(),
        size_of::<Result<(Array, Array), Exception>>(),
        size_of::<Result<(), Exception>>(),
        size_of::<Option<(i64, i64)>>(),
        size_of::<TransientRootsProjection>(),
        OriginalScopeObserver::control_bytes()?,
        super::failure_control_bytes()?,
        crate::backend::runtime::cache::value_completion_control_bytes(2)?,
        crate::backend::runtime::cache::kv::PagedKeyValueCache::original_visible_control_bytes()?,
    ];
    let base = fixed
        .into_iter()
        .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)?;
    let acquisition = [
        size_of::<(&mut OriginalPagedVisibleClaim<'_, '_>, &Stream)>(),
        size_of::<OriginalPagedBlockSource<'_, '_>>(),
        size_of::<Result<Option<usize>, Exception>>(),
        size_of::<Result<(usize, crate::backend::runtime::cache::residency::PinnedCacheBlockLease), Exception>>(),
        size_of::<Result<crate::backend::runtime::cache::residency::PinnedCacheBlockLease, CacheSourceError>>(),
        size_of::<std::slice::Iter<'_, super::super::scan_program::ScanSourceRow>>(),
        crate::backend::runtime::cache::residency::PinnedCacheBlock::fixed_controls()?,
        crate::backend::runtime::cache::residency::CacheResidencyManager::original_scan_control_bytes()?,
        ProjectedPagedSource::promotion_access_control_bytes()?,
        crate::backend::runtime::cache::value_completion_control_bytes(2)?,
    ];
    let per_row = acquisition
        .into_iter()
        .try_fold(std::mem::size_of_val(&acquisition), usize::checked_add)?;
    let slicing = [
        size_of::<(
            &mut OriginalPagedVisibleClaim<'_, '_>,
            usize,
            std::ops::Range<i32>,
            &Stream,
        )>(),
        size_of::<Result<[&Array; 2], Exception>>(),
        size_of::<Option<[&Array; 2]>>(),
        size_of::<(Array, Array)>(),
        size_of::<Result<Array, Exception>>().checked_mul(2)?,
        safemlx::ops::indexing::inline_basic_index_control_bytes()?.checked_mul(2)?,
    ];
    let per_part = slicing
        .into_iter()
        .try_fold(std::mem::size_of_val(&slicing), usize::checked_add)?;
    // Exactly two old-window concatenations only when multiple parts, and two
    // joins with current input only when that old window exists.
    let concatenations = usize::from(parts > 1)
        .checked_add(usize::from(history))?
        .checked_mul(2)?;
    base.checked_add(per_row.checked_mul(rows)?)?
        .checked_add(per_part.checked_mul(parts)?)?
        .checked_add(safemlx::ops::concatenate_axis_control_bytes()?.checked_mul(concatenations)?)
}
