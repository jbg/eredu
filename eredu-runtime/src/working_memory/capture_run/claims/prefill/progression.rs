//! Closed host coupling. Private until architecture causal/readout activation.
use super::*;
use crate::capture::{
    CapturePrefillHookDecision, CapturePrefillObservationPolicy, CapturePrefillProgressError,
};
use crate::prefill::PrefillChunk;
fn progress_error(error: CapturePrefillProgressError) -> CaptureRunHostError {
    CapturePrefillHostError::from(error).into()
}
impl<'a> CaptureStepClaim<'a> {
    pub(crate) fn prepare_prefill_with_progression(
        self,
        inference: InferenceGeometry,
    ) -> Result<ScheduledCaptureStep<'a>, CaptureRunHostError> {
        // This private call provides no architecture semantics. Its future
        // closed caller must carry the bound causal/readout companion.
        let mut step=self.prepare_prefill(inference)?;
        step.initialize_prefill_progression()?;
        Ok(step)
    }
}
impl ScheduledCaptureStep<'_> {
    fn initialize_prefill_progression(&mut self)->Result<(),CaptureRunHostError> {
        let inference=self.frame.prefill.as_ref().ok_or(CapturePrefillHostError::Identity)?.inference;
        let policy=CapturePrefillObservationPolicy::new(self.claim.source,inference).map_err(progress_error)?;
        for (index,slot) in self.frame.prefill.as_mut().expect("validated target source").slots.iter_mut().enumerate() {
            if slot.progression.is_some(){return Err(CapturePrefillHostError::Identity.into());}
            slot.progression=Some(policy.row(index).map_err(progress_error)?.initial_progress());
        }
        self.claim.custody.validate()?;
        Ok(())
    }
    /// The already allocated original companion gains the same finite target
    /// headers. Repeated epochs retain its progressed state and allocate nothing.
    pub(crate) fn prepare_existing_prefill_with_progression(&mut self,inference:InferenceGeometry)->Result<(),CaptureRunHostError> {
        self.claim.custody.validate()?;
        if let Some(actual)=self.frame.prefill.as_ref() {
            if actual.inference!=inference || actual.slots.iter().any(|slot|slot.progression.is_none()) {
                return Err(CapturePrefillHostError::Identity.into());
            }
            return Ok(());
        }
        let captures=self.claim.validate_prefill_source(inference)?;
        self.initialize_prefill_targets(inference,captures)?;
        self.initialize_prefill_progression()
    }
    /// Actual already initialized ordinary schedule; no inferred chunk geometry.
    pub(crate) fn prefill_geometry(&self)->Option<InferenceGeometry> {
        self.frame.prefill.as_ref().map(|targets|targets.inference)
    }
    pub(crate) fn begin_prefill_hook(
        &mut self,
        index: usize,
        chunk: &PrefillChunk,
        path: &str,
    ) -> Result<CapturePrefillHookDecision, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.begin_hook(progress, chunk, path)
            .map_err(progress_error)
    }
    pub(crate) fn begin_routed_prefill_batch(
        &mut self, index: usize, chunk: &PrefillChunk, path: &str,
    ) -> Result<CapturePrefillHookDecision, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self.frame.prefill.as_mut().ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets.slots.get_mut(index).and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.begin_routed_batch(progress, chunk, path).map_err(progress_error)
    }
    /// The same supplied full usage reaches the existing ledger and frame once.
    /// No scalar here affects the original physical H/native reservation.
    pub(crate) fn reserve_prefill_hook(
        &mut self,
        index: usize,
        ledger: &mut dyn eredu_core::capture::CaptureReservation,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<Option<CaptureSkipReason>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        let skipped = row
            .reserve_first(progress, ledger, usage)
            .map_err(progress_error)?;
        let result = if let Some(reason) = &skipped {
            self.record_skip(index, reason.clone(), Some(dtype), CaptureUsage::default())
        } else if row.terminal() {
            self.frame
                .charge_prefill_target(index, dtype, usage)
                .map_err(CaptureRunHostError::from)
        } else {
            self.begin_prefill_target(index, dtype, usage)
        };
        if let Err(error) = result {
            self.fail_prefill_hook(index);
            return Err(error);
        }
        Ok(skipped)
    }
    pub(crate) fn finish_prefill_hook(
        &mut self,
        index: usize,
        fragment: &CapturePrefillFragment<'_, '_>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.finish_hook(progress, fragment).map_err(progress_error)
    }
    pub(crate) fn finish_candidate_prefill_hook(
        &mut self,
        index: usize,
        chunk: &PrefillChunk,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.finish_candidate_hook(progress, chunk)
            .map_err(progress_error)
    }
    pub(crate) fn finish_token_scores_prefill_hook(
        &mut self,
        index: usize,
        chunk: &PrefillChunk,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        let row = policy.row(index).map_err(progress_error)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.finish_token_scores_hook(progress, chunk)
            .map_err(progress_error)
    }
    pub(crate) fn fail_prefill_hook(&mut self, index: usize) {
        if let Some(slot) = self
            .frame
            .prefill
            .as_mut()
            .and_then(|targets| targets.slots.get_mut(index))
        {
            if let Some(progress) = slot.progression.as_mut() {
                progress.fail_hook();
            }
            slot.state = TargetState::Failed;
        }
    }
    pub(super) fn validate_prefill_progression(
        &self,
        chunk: u64,
    ) -> Result<(), CaptureRunHostError> {
        let targets = self
            .frame
            .prefill
            .as_ref()
            .ok_or(CapturePrefillHostError::Identity)?;
        if targets.slots.iter().all(|s| s.progression.is_none()) {
            return Ok(());
        }
        let policy = CapturePrefillObservationPolicy::new(self.claim.source, targets.inference)
            .map_err(progress_error)?;
        for (index, slot) in targets.slots.iter().enumerate() {
            let progress = slot
                .progression
                .as_ref()
                .ok_or(CapturePrefillHostError::Identity)?;
            policy
                .row(index)
                .map_err(progress_error)?
                .validate_chunk_end(progress, chunk)
                .map_err(progress_error)?;
        }
        Ok(())
    }
    pub(super) fn validate_prefill_progression_finished(&self) -> Result<(), CaptureRunHostError> {
        if let Some(targets) = self.frame.prefill.as_ref() {
            for (index, slot) in targets.slots.iter().enumerate() {
                if slot
                    .progression
                    .as_ref()
                    .is_some_and(|progress| !progress.finished())
                {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
            }
        }
        Ok(())
    }
}
