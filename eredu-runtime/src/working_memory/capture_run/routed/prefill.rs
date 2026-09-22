//! One persistent sparse destination across canonical chunks; short exclusive writers.
use super::*;
use crate::working_memory::capture_tensor::prefill::{TargetSlot, TargetState};
impl<'a> ScheduledCaptureStep<'a> {
    /// Borrow the real next sparse fragment after its whole logical target was charged.
    pub fn take_prefill_routed_fragment<'t, 'f, 'p, 'g>(
        &'t mut self,
        index: usize,
        fragment: &'f CaptureRoutedPrefillFragment<'p, 'g>,
    ) -> Result<CaptureRoutedPrefillWriter<'t, 'f, 'p, 'g>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let geometry = fragment.plan().geometry();
        if !std::ptr::eq(geometry.admission(), self.claim.source.admission())
            || geometry.selection_index() != index
            || fragment.plan().inference_geometry() != targets.inference
            || fragment.chunk_index() != targets.next
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Spent)
        {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if slot.done || !matches!(slot.state, TargetState::Claimed | TargetState::Active) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        // The issued fragment cannot be retried after allocation or producer failure.
        slot.state = TargetState::Failed;
        if slot.routed.is_none() {
            let geometry = CaptureRoutedUnitsGeometry::prepare(
                self.claim.source.admission(),
                index,
                CapturePhase::Prefill,
                0,
                None,
            )
            .map_err(CaptureStepError::from)?;
            let plan = CaptureRoutedHostPlan::prepare(geometry)?;
            slot.routed = Some(OwnedCaptureRoutedUnits::allocate(
                &plan,
                claims::ReceiptIdentity {
                    phase: CapturePhase::Prefill,
                    prediction: 0,
                    index,
                    custody: self.claim.custody.share_scheduled(),
                },
            )?);
        }
        slot.state = TargetState::Claimed;
        Ok(CaptureRoutedPrefillWriter {
            slot,
            fragment,
            covered: 0,
            finished: false,
        })
    }
    /// Shared semantic progression acknowledges only a completed physical sparse writer.
    pub(crate) fn finish_routed_prefill_hook(
        &mut self,
        index: usize,
        fragment: &CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<(), CaptureRunHostError> {
        self.finish_routed_progress(index, fragment, false)
    }
    /// The original partition target has no serial sparse payload; its final
    /// claim remains unavailable until all-rank assembly and delivery succeed.
    pub(crate) fn finish_partition_routed_prefill_hook(
        &mut self,
        index: usize,
        fragment: &CaptureRoutedPrefillFragment<'_, '_>,
    ) -> Result<(), CaptureRunHostError> {
        self.finish_routed_progress(index, fragment, true)
    }
    fn finish_routed_progress(
        &mut self,
        index: usize,
        fragment: &CaptureRoutedPrefillFragment<'_, '_>,
        partition: bool,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if if partition {
            slot.done || slot.state != TargetState::Assembling || slot.routed.is_some()
        } else {
            !slot.done || slot.state != TargetState::Active || slot.routed.is_none()
        } {
            return Err(CapturePrefillHostError::Incomplete { index }.into());
        }
        let policy = crate::capture::CapturePrefillObservationPolicy::new(
            self.claim.source,
            targets.inference,
        )
        .map_err(CapturePrefillHostError::from)?;
        let row = policy.row(index).map_err(CapturePrefillHostError::from)?;
        let progress = slot
            .progression
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        row.finish_routed_hook(progress, fragment)
            .map_err(CapturePrefillHostError::from)?;
        Ok(())
    }
}
/// No owning payload escapes: failures and drops leave initialized rows in the frame.
#[derive(Debug)]
pub struct CaptureRoutedPrefillWriter<'t, 'f, 'p, 'a> {
    slot: &'t mut TargetSlot,
    fragment: &'f CaptureRoutedPrefillFragment<'p, 'a>,
    covered: u64,
    finished: bool,
}
impl CaptureRoutedPrefillWriter<'_, '_, '_, '_> {
    /// Actual immutable fragment; it carries no native grant.
    pub fn fragment(&self) -> &CaptureRoutedPrefillFragment<'_, '_> {
        self.fragment
    }
    /// Check the exact original account before a source-bound native producer.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.slot
            .routed
            .as_ref()
            .expect("initialized routed target")
            .identity
            .custody
            .validate_scheduled_native(native)
    }
    /// Start one selected route using its actual physical flattened token.
    pub fn begin_row(
        &mut self,
        physical_token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        let owner = self
            .slot
            .routed
            .as_mut()
            .expect("initialized routed target");
        if !self.fragment.selects(physical_token, slot) {
            return owner.remember(Err(RoutedUnitValidationError::Selection.into()));
        }
        owner.begin_row(
            None,
            self.fragment
                .logical_token(physical_token)
                .expect("selected token"),
            slot,
            expert,
            coefficient,
        )
    }
    /// Copy one scalar to the retained fixed row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.slot
            .routed
            .as_mut()
            .expect("initialized routed target")
            .push_f32(value)
    }
    /// Complete a filled row without creating another destination.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.slot
            .routed
            .as_mut()
            .expect("initialized routed target")
            .finish_row()
    }
    /// Record only a visited provider range, splitting flattened batch boundaries.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        let owner = self
            .slot
            .routed
            .as_mut()
            .expect("initialized routed target");
        let Some(ranges) = self.fragment.source_ranges(start, end) else {
            return owner.remember(Err(RoutedUnitValidationError::Chunks.into()));
        };
        for [lo, hi] in ranges {
            owner.source_chunk(lo, hi)?;
        }
        self.covered = self
            .covered
            .checked_add(end - start)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(())
    }
    /// Finish one real provider batch. The same chunk accepts subsequent disjoint
    /// batches until all its physical tokens are covered; the target seals later.
    pub fn finish(mut self) -> Result<(), CaptureRoutedHostError> {
        let owner = self
            .slot
            .routed
            .as_mut()
            .expect("initialized routed target");
        owner.check()?;
        let total = self
            .slot
            .routed_covered
            .checked_add(self.covered)
            .filter(|n| *n <= self.fragment.source_tokens());
        if owner.data.active.is_some() || self.covered == 0 || total.is_none() {
            return owner.remember(Err(RoutedUnitValidationError::Incomplete.into()));
        }
        self.slot.routed_covered = total.expect("checked source coverage");
        self.slot.state = TargetState::Active;
        self.slot.done = self.slot.routed_covered == self.fragment.source_tokens();
        self.finished = true;
        Ok(())
    }
}
impl Drop for CaptureRoutedPrefillWriter<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.slot.state = TargetState::Failed;
        }
    }
}

/// Exact stamped source binding for a short sparse writer. The source remains
/// in the original scope after this loan ends, including failed partial copies.
pub struct CaptureRoutedPrefillTransfer<'t, 'f, 'p, 'a, 's, K: Ord + Send + 'static> {
    writer: CaptureRoutedPrefillWriter<'t, 'f, 'p, 'a>,
    source: WorkingMemoryStorage<K>,
    _segment: &'s mut crate::working_memory::CaptureSourceSegment,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<'t, 'f, 'p, 'a> CaptureRoutedPrefillWriter<'t, 'f, 'p, 'a> {
    /// Authenticate this actual chunk and pin all original native inputs before
    /// reading their scalars. This grants no completion or allocation authority.
    pub fn prepare_with_segment_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        segment: &'s mut crate::working_memory::CaptureSourceSegment,
        source: WorkingMemoryStorage<K>,
    ) -> Result<CaptureRoutedPrefillTransfer<'t, 'f, 'p, 'a, 's, K>, CaptureRunHostError> {
        segment.validate_routed_prefill_fragment(self.fragment)?;
        let rollback = self
            .slot
            .routed
            .as_ref()
            .expect("initialized routed target")
            .identity
            .custody
            .bind_segment_source(native, segment, &source)?;
        Ok(CaptureRoutedPrefillTransfer {
            writer: self,
            source,
            _segment: segment,
            native: rollback.commit(),
        })
    }
}
impl<K: Ord + Send + 'static> CaptureRoutedPrefillTransfer<'_, '_, '_, '_, '_, K> {
    /// Same immutable mapping as the issued original target.
    pub fn fragment(&self) -> &CaptureRoutedPrefillFragment<'_, '_> {
        self.writer.fragment()
    }
    /// Revalidate the original source origins and account around each copy.
    pub fn validate(&self) -> Result<(), CaptureRunHostError> {
        self.writer
            .slot
            .routed
            .as_ref()
            .expect("initialized routed target")
            .identity
            .custody
            .validate_transfer(self.native, &self.source)?;
        if self.writer.slot.state == TargetState::Failed {
            return Err(CapturePrefillHostError::Identity.into());
        }
        Ok(())
    }
    /// Begin an already selected physical route; no row is allocated.
    pub fn begin_row(
        &mut self,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.begin_row(token, slot, expert, coefficient)?;
        Ok(())
    }
    /// Copy one scalar into the paid row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.push_f32(value)?;
        Ok(())
    }
    /// Seal the initialized row.
    pub fn finish_row(&mut self) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.finish_row()?;
        Ok(())
    }
    /// Preserve the actual visited provider interval.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.source_chunk(start, end)?;
        Ok(())
    }
    /// Finish only the host copy, retaining every native obligation.
    pub fn finish(self) -> Result<(), CaptureRunHostError> {
        self.validate()?;
        self.writer.finish()?;
        Ok(())
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for CaptureRoutedPrefillTransfer<'_, '_, '_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureRoutedPrefillTransfer")
            .field("writer", &self.writer)
            .finish_non_exhaustive()
    }
}
