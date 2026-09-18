//! Whole-lifetime tracking storage for the selected resident fixed Eval producer.
use super::*;
use eredu_runtime::working_memory::{SamplingWorkspacePhase, WorkingMemoryError};
use std::num::NonZeroU64;

/// Necessary partial facts remain available for diagnostics. Only full_capacity
/// certifies the complete selected resident Record population in a fresh arena.
/// It says nothing about Graph/payload/queue or host/disk producer completeness.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ResidentRecordStorage {
    pub(crate) minimum_capacity: u64,
    pub(crate) known_constructor_bytes: u64,
    pub(crate) full_capacity: Option<u64>,
}
impl ResidentRecordStorage {
    fn include_sampling(&mut self, rows: &[ResidentSamplingRecipe], extents: &mut usize) -> Result<bool, crate::backend::error::Error> {
        let error = crate::backend::error::Error::PrefillControl;
        let mut complete = true;
        for row in rows {
            match row.phase() {
                SamplingWorkspacePhase::Preparation => {
                    // Only an observed empty trace or the closed eager U32 key
                    // constructor has no native evaluation/Record submission.
                    complete &= row.preparation.is_some() && row.completion().is_none();
                }
                SamplingWorkspacePhase::Step { .. } => match row.completion() {
                    Some(completion) => {
                        self
                            .include(completion.traversal, extents)
                            .map_err(error)?;
                        self
                            .include_nested(
                                completion.traversal,
                                NestedCompletionRoots::uniform(completion.nested_completions, completion.nested_root_capacity.max(3)),
                                extents,
                            )
                            .map_err(error)?;
                    }
                    None => complete = false,
                },
            }
        }
        Ok(complete)
    }
    fn finish_capacity(&mut self, complete: bool, extents: usize) -> Result<(), crate::backend::error::Error> {
        if complete {
            let capacity = safemlx::SubmissionRecordQuota::fresh_capacity_for_extents(extents)
                .ok_or(crate::backend::error::Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            self.full_capacity = Some(u64::try_from(capacity)
                .map_err(|_| crate::backend::error::Error::PrefillControl(WorkingMemoryError::Overflow))?);
        }
        Ok(())
    }
    fn include_source_copies(&mut self, copies: host_copies::HostCopies,
        transfers: host_copies::HostTransfers, forwards: usize, extents: &mut usize)
        -> Result<(), WorkingMemoryError> {
        let overflow = || WorkingMemoryError::Overflow;
        for (traversal, attempts) in [(copies.traversal, copies.per_forward),
            (copies.aggregate_traversal, transfers.per_forward)] {
            let attempts = attempts.checked_mul(forwards).ok_or_else(overflow)?;
            let mut one = Self::default();
            let mut one_extent = 0;
            one.include(traversal, &mut one_extent)?;
            self.minimum_capacity = self.minimum_capacity.max(one.minimum_capacity);
            self.known_constructor_bytes = self.known_constructor_bytes.checked_add(
                one.known_constructor_bytes.checked_mul(u64::try_from(attempts).map_err(|_|overflow())?)
                    .ok_or_else(overflow)?).ok_or_else(overflow)?;
            *extents = extents.checked_add(one_extent.checked_mul(attempts).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let waits = safemlx::OperationEvent::wait_record_layout(
            transfers.waits_per_forward.checked_mul(forwards).ok_or_else(overflow)?)
            .ok_or(WorkingMemoryError::UnknownBound)?;
        *extents = extents.checked_add(waits.record_allocation_extents()
            .ok_or(WorkingMemoryError::UnknownBound)?).ok_or_else(overflow)?;
        self.known_constructor_bytes = self.known_constructor_bytes.checked_add(
            u64::try_from(waits.total_record_requested_bytes()).map_err(|_|overflow())?)
            .ok_or_else(overflow)?;
        Ok(())
    }
    /// Same fixed Eval constructors for one non-text numerical cut.
    pub(super) fn for_completion(completion: ResidentCompletionRecipe) -> Option<Self> {
        Self::for_completion_with_waits(completion,0)
    }
    pub(super) fn for_completion_with_waits(completion:ResidentCompletionRecipe,waits:usize)->Option<Self> {
        Self::for_completion_sources(completion, waits, None)
    }
    pub(super) fn for_completion_sources(completion: ResidentCompletionRecipe, waits: usize,
        sources: Option<host_copies::PreparedSourceCopies>) -> Option<Self> {
        let mut value = Self::default();
        let mut extents = 0;
        value.include(completion.traversal, &mut extents).ok()?;
        value
            .include_nested(
                completion.traversal,
                NestedCompletionRoots::uniform(completion.nested_completions, completion.nested_root_capacity.max(3)),
                &mut extents,
            )
            .ok()?;
        if waits!=0 {
            let source=safemlx::OperationEvent::wait_record_layout(waits)?;
            extents=extents.checked_add(source.record_allocation_extents()?)?;
            value.known_constructor_bytes=value.known_constructor_bytes.checked_add(
                u64::try_from(source.total_record_requested_bytes()).ok()?)?;
        }
        if let Some(source) = sources {
            value.include_source_copies(source.copies, source.transfers, 1, &mut extents).ok()?;
        }
        value.full_capacity = Some(
            u64::try_from(safemlx::SubmissionRecordQuota::fresh_capacity_for_extents(
                extents,
            )?)
            .ok()?,
        );
        Some(value)
    }
    fn include(
        &mut self,
        traversal: safemlx::OperationEvalTraversalLayout,
        extents: &mut usize,
    ) -> Result<(), WorkingMemoryError> {
        let bytes = traversal
            .minimum_record_capacity()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let bytes = u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)?;
        self.minimum_capacity = self.minimum_capacity.max(bytes);
        self.known_constructor_bytes = self
            .known_constructor_bytes
            .checked_add(bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        *extents = extents
            .checked_add(
                traversal
                    .record_allocation_extents()
                    .ok_or(WorkingMemoryError::UnknownBound)?,
            )
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(())
    }
    fn include_nested(
        &mut self, traversal: safemlx::OperationEvalTraversalLayout,
        roots: NestedCompletionRoots<'_>, extents: &mut usize,
    ) -> Result<(), WorkingMemoryError> {
        for count in roots.iter() {
            self.include(
                nested_traversal_with_roots(traversal, count)
                    .ok_or(WorkingMemoryError::UnknownBound)?,
                extents,
            )?;
        }
        Ok(())
    }
    pub(crate) fn validate_capacity(self, configured_bytes: u64) -> Result<(), WorkingMemoryError> {
        let required_bytes = self.full_capacity.unwrap_or(self.minimum_capacity);
        if configured_bytes < required_bytes {
            return Err(WorkingMemoryError::SubmissionTrackingCapacity {
                required_bytes,
                configured_bytes,
            });
        }
        Ok(())
    }
    pub(crate) fn select_capacity(
        self,
        ceiling: NonZeroU64,
    ) -> Result<NonZeroU64, WorkingMemoryError> {
        self.select_for_policy(Some(ceiling))
    }
    pub(crate) fn select_for_policy(
        self,
        ceiling: Option<NonZeroU64>,
    ) -> Result<NonZeroU64, WorkingMemoryError> {
        if let Some(ceiling) = ceiling {
            self.validate_capacity(ceiling.get())?;
        }
        self.full_capacity
            .and_then(NonZeroU64::new)
            .ok_or(WorkingMemoryError::UnknownBound)
    }
}
impl ResidentNativeRecipe {
    pub(crate) fn record_storage_requirement(
        &self,
    ) -> Result<ResidentRecordStorage, crate::backend::error::Error> {
        self.record_storage_requirement_with_rows(self.records())
    }
    pub(super) fn record_storage_requirement_with_rows(
        &self,
        records: &[ResidentSpanRecipe],
    ) -> Result<ResidentRecordStorage, crate::backend::error::Error> {
        let error = crate::backend::error::Error::PrefillControl;
        let mut requirement = ResidentRecordStorage::default();
        let mut extents = 0usize;
        if let Some(copy) = self.resume_copy {
            let layout = copy.layout();
            requirement.minimum_capacity = requirement.minimum_capacity.max(
                u64::try_from(
                    layout
                        .traversal
                        .minimum_record_capacity()
                        .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?,
                )
                .map_err(|_| error(WorkingMemoryError::Overflow))?,
            );
            requirement.known_constructor_bytes = u64::try_from(
                layout
                    .traversal
                    .minimum_record_capacity()
                    .and_then(|n| n.checked_mul(layout.completion_attempts))
                    .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
            )
            .map_err(|_| error(WorkingMemoryError::Overflow))?;
            extents = layout.record_extents;
        }
        let mut complete = !records.is_empty() || self.is_terminal_resume();
        for row in records {
            match row.traversal() {
                Some(traversal) => {
                    requirement
                        .include(traversal, &mut extents)
                        .map_err(error)?;
                    requirement
                        .include_nested(traversal, NestedCompletionRoots::for_row(row), &mut extents)
                        .map_err(error)?;
                }
                None => complete = false,
            }
        }
        if let Some(copies) = self.host_copies {
            requirement.include_source_copies(copies,
                self.host_transfers.ok_or_else(|| error(WorkingMemoryError::UnknownBound))?,
                records.len(), &mut extents).map_err(error)?;
        }
        // The accepted group bank owns a finite consumer population in each
        // equation role. Each wait allocates the owning native WaitRecord and
        // its capture/stream receipt buffers, including failed prefixes.
        if self.neural_waits_per_forward() != 0 {
            let waits =
                safemlx::OperationEvent::wait_record_layout(self.neural_waits_per_forward())
                    .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?;
            let one = waits
                .record_allocation_extents()
                .ok_or_else(|| error(WorkingMemoryError::UnknownBound))?;
            extents = extents
                .checked_add(
                    one.checked_mul(records.len())
                        .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
                )
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
            requirement.known_constructor_bytes = requirement
                .known_constructor_bytes
                .checked_add(
                    u64::try_from(waits.total_record_requested_bytes())
                        .ok()
                        .and_then(|n| n.checked_mul(records.len() as u64))
                        .ok_or_else(|| error(WorkingMemoryError::Overflow))?,
                )
                .ok_or_else(|| error(WorkingMemoryError::Overflow))?;
        }
        // finish() already authenticates the exact plan, every equation row,
        // and Preparation followed by one sampling row per output attempt.
        complete &= requirement.include_sampling(self.sampling_records(), &mut extents)?;
            // Each Roots/ModelExecution collector is consumed once, even on
            // constructor failure. SamplingEvent has the same submitted-once
            // guard. Sum all attempted constructors: cancellation/failed prefixes
            // can only remove suffixes, and no early retirement credit is used.
            // A true nested block has its own fixed Eval attempt. Its enclosing
            // row ceiling is repeated conservatively for each configured call;
            // actual per-block retirement still occurs in the shared worker.
            // Resident completion and scalar reads query/yield existing records;
            // bounded-policy consumer WaitRecords were added separately above.
            // Cross-stream Event/Fence
            // operations execute inside the counted Fixed Eval Record.
        requirement.finish_capacity(complete, extents)?;
        Ok(requirement)
    }
}

impl ResidentSamplingProgram {
    pub(crate) fn record_storage_requirement(&self) -> Result<ResidentRecordStorage, crate::backend::error::Error> {
        let mut required = ResidentRecordStorage::default();
        let mut extents = 0;
        let complete = required.include_sampling(self.rows(), &mut extents)?;
        required.finish_capacity(complete, extents)?;
        Ok(required)
    }
}
