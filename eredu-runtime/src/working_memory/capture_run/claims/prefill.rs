//! Persistent one-buffer targets; short fragment borrows never enter the bank.
use super::*;
pub use crate::working_memory::capture_tensor::prefill::CapturePrefillHostError;
use crate::working_memory::capture_tensor::prefill::{
    OwnedPrefillTensor, PrefillTargets, TargetSlot, TargetState,
};
use eredu_core::{InferenceGeometry, TensorObservation, TensorObservationData};

mod progression;
mod partition;
mod transfer;
pub use transfer::CapturePrefillFragmentTransfer;

impl<'a> CaptureStepClaim<'a> {
    /// Prepare p0's fixed partial-target slots under the original whole-run H.
    /// Only row-wise ordinary geometry is accepted. This authenticates no actual
    /// activation or native execution, and changes no managed applicability gate.
    pub fn prepare_prefill(
        self,
        inference: InferenceGeometry,
    ) -> Result<ScheduledCaptureStep<'a>, CaptureRunHostError> {
        let captures=self.validate_prefill_source(inference)?;
        let mut step=self.prepare()?;
        step.initialize_prefill_targets(inference,captures)?;
        Ok(step)
    }
    fn validate_prefill_source(&self,inference:InferenceGeometry)->Result<usize,CaptureRunHostError> {
        self.custody.validate()?;
        if self.phase != CapturePhase::Prefill
            || self.prediction != 0
        {
            return Err(CapturePrefillHostError::Identity.into());
        }
        // Inference reserves the full F + P + M frontier, even when p0 has no
        // active targets. Capture's last actual coordinate only needs M - 1.
        inference
            .cached_positions
            .checked_add(inference.input_positions)
            .and_then(|n| n.checked_add(inference.max_output_tokens))
            .ok_or(CapturePrefillHostError::from(
                CapturePrefillGeometryError::Overflow,
            ))?;
        // Validate every eligible geometry before allocating even the slot box.
        let captures=self.source.admission().plan().selections.len();
        for (index, state) in self.row[1..1+captures].iter().enumerate() {
            if *state == ClaimState::Available {
                if matches!(
                    self.source.admission().plan().selections[index].transform,
                    CaptureTransform::TopCandidates { .. }
                ) {
                    CaptureCandidateGeometry::prepare(
                        self.source.admission(),
                        index,
                        self.phase,
                        self.prediction,
                        None,
                    )
                    .map_err(CaptureStepError::from)?;
                } else if matches!(
                    self.source.admission().plan().selections[index].transform,
                    CaptureTransform::TokenScores { .. }
                ) {
                    CaptureTokenScoreGeometry::prepare(
                        self.source.admission(),
                        index,
                        self.phase,
                        self.prediction,
                        None,
                    )
                    .map_err(CaptureStepError::from)?;
                } else if matches!(
                    self.source.admission().plan().selections[index].transform, CaptureTransform::RoutedUnits
                ) {
                    CaptureRoutedPrefillPlan::prepare(self.source.admission(),index,inference)
                        .map_err(CapturePrefillHostError::from)?;
                } else if matches!(
                    self.source.admission().plan().selections[index].transform,
                    CaptureTransform::Summary | CaptureTransform::Histogram { .. }
                ) {
                    CapturePrefillTransformPlan::prepare(self.source.admission(), index, inference)
                        .map_err(CaptureStepError::from)?;
                } else {
                    CapturePrefillRowAssembly::prepare(self.source.admission(), index, inference)
                        .map_err(CapturePrefillHostError::from)?;
                }
            }
        }
        // The no-active-target case retains the complete prompt geometry and
        // rejects output caps beyond the immutable source. A saved p0 prefix
        // uses the same fixed target/fragment worker.
        let request = self.source.admission().request();
        if inference.input_positions == 0
            || inference.prefill_chunk_positions == 0
            || inference.prefill_chunk_positions > inference.input_positions
            || request.batch != inference.batch_size
            || request.prompt_tokens != inference.input_positions
            || inference.max_output_tokens > request.max_predictions
            || self
                .source
                .admission()
                .text_origin()
                .map(|o| o.cached_positions)
                != Some(inference.cached_positions)
        {
            return Err(CapturePrefillHostError::Identity.into());
        }
        Ok(captures)
    }
}
impl ScheduledCaptureStep<'_> {
    fn initialize_prefill_targets(&mut self,inference:InferenceGeometry,captures:usize)->Result<(),CaptureRunHostError> {
        if self.frame.prefill.is_some(){return Err(CapturePrefillHostError::Identity.into());}
        let mut slots = Vec::with_capacity(captures);
        for state in &self.claim.row[1..1+captures] {
            slots.push(TargetSlot {
                tensor: None,
                completed: None,
                summary: None,
                histogram: None,
                routed: None,
                routed_covered: 0,
                state: if *state == ClaimState::Available {
                    TargetState::Unstarted
                } else {
                    TargetState::Inactive
                },
                dtype: None,
                done: false,
                progression: None,
            });
        }
        self.frame.prefill = Some(PrefillTargets {
            slots: slots.into_boxed_slice(),
            inference,
            next: 0,
        });
        self.claim.custody.validate()?;
        Ok(())
    }
}
impl ScheduledCaptureStep<'_> {
    pub(super) fn retire_prefill_target(&mut self, index: usize) {
        if let Some(targets) = self.frame.prefill.as_mut() {
            if let Some(target) = targets.slots.get_mut(index) {
                target.state = TargetState::Inactive;
            }
        }
    }

    /// Spend this logical full-target claim once and record its already-reserved
    /// full-value usage. Fragment writes never add/refund logical usage or H.
    pub fn begin_prefill_target(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        additional: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_ref()
            .ok_or(CapturePrefillHostError::Identity)?;
        if !targets
            .slots
            .get(index)
            .is_some_and(|s| s.state == TargetState::Unstarted)
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Available)
        {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        self.frame
            .charge_prefill_target(index, dtype.clone(), additional)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        let slot = &mut self.frame.prefill.as_mut().expect("checked prefill").slots[index];
        slot.dtype = Some(dtype);
        slot.state = TargetState::Claimed;
        Ok(())
    }
    /// Borrow one target for the exact next derived fragment. Neither source
    /// offsets nor mutable destination slices are accepted or exported.
    pub fn take_prefill_fragment<'t, 'f, 'p, 'a>(
        &'t mut self,
        index: usize,
        fragment: &'f CapturePrefillFragment<'p, 'a>,
    ) -> Result<CapturePrefillFragmentClaim<'t, 'f, 'p, 'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let geometry = fragment.assembly().logical_geometry();
        if !std::ptr::eq(geometry.admission(), self.claim.source.admission())
            || geometry.selection_index() != index
            || fragment.assembly().inference_geometry() != targets.inference
        {
            return Err(CapturePrefillHostError::Identity.into());
        }
        if fragment.chunk_index() != targets.next {
            return Err(CapturePrefillHostError::Order.into());
        }
        let slot = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        if slot.done || !matches!(slot.state, TargetState::Claimed | TargetState::Active) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        Ok(CapturePrefillFragmentClaim {
            slot: Some(slot),
            fragment,
            index,
            custody: self.claim.custody.share_scheduled(),
            partition: false,
        })
    }
    /// Commit host coverage of one canonical chunk. Zero-contribution fragments
    /// require no writer; a zero-sized target still needs an actual initial claim
    /// and finished empty writer. No native completion or pin release occurs.
    pub fn complete_prefill_chunk(&mut self, chunk_index: u64) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let count = targets
            .inference
            .input_positions
            .div_ceil(targets.inference.prefill_chunk_positions);
        if targets.next != chunk_index || chunk_index >= count {
            return Err(CapturePrefillHostError::Order.into());
        }
        // Preflight shared semantic hooks before any physical cursor changes.
        self.validate_prefill_progression(chunk_index)?;
        let targets = self.frame.prefill.as_mut().expect("checked prefill");
        // Preflight the entire row: a later missing target cannot advance earlier ones.
        for (index, slot) in targets.slots.iter().enumerate() {
            if slot.state == TargetState::Inactive {
                continue;
            }
            if matches!(slot.state, TargetState::Remote | TargetState::Assembling) {
                // The global target has no local payload: source terms are remote
                // or remain in the separately paid partition fragment bank.
                // Logical hook progression above still authenticates this exact
                // committed chunk, including zero-contribution occurrences.
                if slot.tensor.is_some() || slot.completed.is_some() || slot.summary.is_some() || slot.histogram.is_some() || slot.routed.is_some() || slot.done {
                    return Err(CapturePrefillHostError::Identity.into());
                }
                continue;
            }
            if matches!(
                self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
            ) {
                if (chunk_index + 1 == count && slot.state != TargetState::Recorded)
                    || (chunk_index + 1 != count && slot.state != TargetState::Unstarted)
                {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            if matches!(self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::RoutedUnits) {
                if slot.state!=TargetState::Active || !slot.done || slot.routed.is_none() {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            if matches!(
                self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::Histogram { .. }
            ) {
                if slot.state != TargetState::Active || !slot.done || slot.histogram.is_none() {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            if matches!(
                self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::Summary
            ) {
                if slot.state != TargetState::Active || !slot.done || slot.summary.is_none() {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            let assembly = CapturePrefillRowAssembly::prepare(
                self.claim.source.admission(),
                index,
                targets.inference,
            )
            .map_err(CapturePrefillHostError::from)?;
            let fragment = assembly
                .fragment(chunk_index)
                .map_err(CapturePrefillHostError::from)?;
            if slot.state == TargetState::Failed
                || slot.state == TargetState::Recorded
                || (fragment.output_elements() > 0 && !slot.done)
                || (chunk_index + 1 == count && slot.tensor.is_none())
            {
                return Err(CapturePrefillHostError::Incomplete { index }.into());
            }
        }
        for slot in &mut targets.slots {
            if let Some(progress) = slot.progression.as_mut() {
                progress.advance_preflighted();
            }
            slot.done = false;
            slot.routed_covered = 0;
        }
        targets.next += 1;
        Ok(())
    }
    /// Seal complete targets into their existing records without a second data
    /// buffer. A health/record rejection keeps the same completed owner in its
    /// private slot; repeated success cannot issue another claim or charge.
    pub fn finish_prefill_targets(&mut self) -> Result<(), CaptureRunHostError> {
        self.finish_prefill_targets_inner(false)
    }
    /// Local producers must seal before their completed receipt can be encoded.
    /// Remote targets remain missing until exact delivery; this cannot seal the
    /// frame or establish native completion/transaction acceptance.
    pub(crate) fn finish_local_prefill_targets(&mut self) -> Result<(), CaptureRunHostError> {
        self.finish_prefill_targets_inner(true)
    }
    fn finish_prefill_targets_inner(&mut self, allow_remote: bool) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        self.validate_prefill_progression_finished()?;
        let targets = self
            .frame
            .prefill
            .as_ref()
            .ok_or(CapturePrefillHostError::Identity)?;
        if targets.next
            != targets
                .inference
                .input_positions
                .div_ceil(targets.inference.prefill_chunk_positions)
        {
            return Err(CapturePrefillHostError::Order.into());
        }
        for (index, slot) in targets.slots.iter().enumerate() {
            if matches!(slot.state, TargetState::Inactive | TargetState::Recorded | TargetState::RemoteRecorded | TargetState::AssemblyRecorded)
                || (allow_remote && matches!(slot.state, TargetState::Remote | TargetState::Assembling))
            {
                continue;
            }
            if matches!(self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::RoutedUnits) {
                if slot.state!=TargetState::Active || slot.routed.is_none() {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            if matches!(
                self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::Histogram { .. }
            ) {
                let geometry = CaptureHistogramGeometry::prepare(
                    self.claim.source.admission(),
                    index,
                    CapturePhase::Prefill,
                    0,
                    None,
                )
                .map_err(CaptureStepError::from)?;
                let value = slot
                    .histogram
                    .as_ref()
                    .filter(|_| slot.state == TargetState::Active)
                    .ok_or(CapturePrefillHostError::Incomplete { index })?;
                crate::capture::reduction::validate_histogram_fixed(
                    value,
                    geometry.edges(),
                    geometry.elements() as u64,
                )
                .map_err(CaptureStepError::from)?;
                continue;
            }
            if matches!(
                self.claim.source.admission().plan().selections[index].transform,
                CaptureTransform::Summary
            ) {
                let geometry = CaptureSummaryGeometry::prepare(
                    self.claim.source.admission(),
                    index,
                    CapturePhase::Prefill,
                    0,
                    None,
                )
                .map_err(CaptureStepError::from)?;
                if slot.state != TargetState::Active
                    || !slot.summary.as_ref().is_some_and(|summary| {
                        summary.value().elements == geometry.elements() as u64
                    })
                {
                    return Err(CapturePrefillHostError::Incomplete { index }.into());
                }
                continue;
            }
            if slot.state != TargetState::Active
                || (slot.completed.is_none()
                    && !slot
                        .tensor
                        .as_ref()
                        .is_some_and(|t| t.covered == t.data.len()))
            {
                return Err(CapturePrefillHostError::Incomplete { index }.into());
            }
        }
        let count = targets.slots.len();
        for index in 0..count {
            self.claim.custody.validate()?;
            let slot = &mut self.frame.prefill.as_mut().expect("checked prefill").slots[index];
            if matches!(slot.state, TargetState::RemoteRecorded | TargetState::AssemblyRecorded) {
                slot.state = TargetState::Recorded;
                continue;
            }
            if matches!(slot.state, TargetState::Inactive | TargetState::Recorded)
                || (allow_remote && matches!(slot.state, TargetState::Remote | TargetState::Assembling))
            {
                continue;
            }
            if let Some(owner)=slot.routed.take() {
                slot.state=TargetState::Failed;
                let dtype=slot.dtype.clone().expect("claimed dtype");
                let receipt=owner.finish().map_err(|failure|CaptureRunHostError::Routed(failure.error().clone()))?;
                self.record_routed_units(receipt,dtype,CaptureUsage::default())?;
                self.frame.prefill.as_mut().expect("checked prefill").slots[index].state=TargetState::Recorded;
                continue;
            }
            if let Some(value) = slot.histogram.take() {
                let dtype = slot.dtype.clone().expect("claimed dtype");
                self.frame
                    .record_histogram(index, dtype, value, CaptureUsage::default())?;
                self.frame.prefill.as_mut().expect("checked prefill").slots[index].state =
                    TargetState::Recorded;
                continue;
            }
            if let Some(summary) = slot.summary.as_ref() {
                let value = summary.value();
                let dtype = slot.dtype.clone().expect("claimed dtype");
                self.frame
                    .record_summary(index, dtype, value, CaptureUsage::default())?;
                let slot = &mut self.frame.prefill.as_mut().expect("checked prefill").slots[index];
                slot.state = TargetState::Recorded;
                continue;
            }
            if slot.completed.is_none() {
                let OwnedPrefillTensor {
                    shape,
                    data,
                    covered: _,
                    custody,
                } = slot.tensor.take().expect("complete target");
                // The closed allocator derived exactly this product. No setter
                // can change shape/length; zero initialization is not coverage.
                let observation = TensorObservation::new(shape, TensorObservationData::F32(data))
                    .expect("checked fixed target shape");
                slot.completed = Some(SharedTensorObservation::retain(observation, custody));
            }
            let tensor = slot.completed.as_ref().expect("completed target").clone();
            let dtype = slot.dtype.clone().expect("claimed dtype");
            // Keep the original shared owner in the slot until this fallible
            // record mutation succeeds, including concurrent parent closure.
            self.frame
                .record_tensor(index, dtype, tensor, CaptureUsage::default())?;
            let slot = &mut self.frame.prefill.as_mut().expect("checked prefill").slots[index];
            slot.completed = None;
            slot.state = TargetState::Recorded;
        }
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn prefill_storage(
        &self,
        index: usize,
    ) -> Option<(*const f32, usize, usize, usize)> {
        let tensor = self
            .frame
            .prefill
            .as_ref()?
            .slots
            .get(index)?
            .tensor
            .as_ref()?;
        Some((
            tensor.data.as_ptr(),
            tensor.data.capacity(),
            tensor.data.len(),
            tensor.covered,
        ))
    }
}

/// Short exclusive borrow of one nonrefundable target and one derived fragment.
/// Dropping it makes the target terminally incomplete, retaining any payload.
#[derive(Debug)]
pub struct CapturePrefillFragmentClaim<'t, 'f, 'p, 'a> {
    slot: Option<&'t mut TargetSlot>,
    fragment: &'f CapturePrefillFragment<'p, 'a>,
    index: usize,
    custody: CaptureTensorCustody,
    partition: bool,
}
impl<'t, 'f, 'p, 'a> CapturePrefillFragmentClaim<'t, 'f, 'p, 'a> {
    // Only the receipt-bound host bank may choose a projected destination.
    // Its retained host plan and exact source comparison precede this loan;
    // ordinary scheduled claims keep their original allocation path.
    pub(in crate::working_memory) fn from_partition(
        slot: &'t mut TargetSlot, fragment: &'f CapturePrefillFragment<'p, 'a>,
        index: usize, custody: CaptureTensorCustody,
    ) -> Self {
        Self { slot: Some(slot), fragment, index, custody, partition: true }
    }

    /// Borrow the exact mapping already bound to this target, source and host
    /// chunk. This creates no native chunk, source or execution authority.
    pub fn fragment(&self) -> &'f CapturePrefillFragment<'p, 'a> {
        self.fragment
    }

    /// Read-only original-account check before a generated producer. The actual
    /// segment/stamp/source and numerical admission remain caller obligations.
    pub fn validate_native_scope(
        &self,
        native: &crate::working_memory::WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.custody.validate_scheduled_native(native)
    }

    /// Allocate/initialize the one full destination on first use, otherwise
    /// retain that exact buffer. This is host construction only, not a source
    /// read, native grant, completion proof or a second reservation.
    pub fn prepare(
        mut self,
    ) -> Result<CapturePrefillFragmentWriter<'t, 'f, 'p, 'a>, CaptureRunHostError> {
        self.custody.validate()?;
        let slot = self.slot.as_mut().expect("owned fragment claim");
        if slot.tensor.is_none() && self.partition {
            slot.tensor = Some(OwnedPrefillTensor::allocate_geometry(
                self.fragment.assembly().logical_geometry(), self.custody.share_scheduled(),
            )?);
        } else if slot.tensor.is_none() {
            let geometry = CaptureTensorGeometry::prepare(
                self.fragment.assembly().logical_geometry().admission(),
                self.index,
                CapturePhase::Prefill,
                0,
                None,
            )
            .map_err(CaptureStepError::from)?;
            let plan = CaptureTensorHostPlan::prepare(geometry)?;
            slot.tensor = Some(OwnedPrefillTensor::allocate(
                plan,
                self.custody.share_scheduled(),
            )?);
        }
        self.custody.validate()?;
        let slot = self.slot.take().expect("owned fragment claim");
        Ok(CapturePrefillFragmentWriter {
            slot,
            fragment: self.fragment,
            index: self.index,
            cursor: 0,
            finished: false,
        })
    }
}
impl Drop for CapturePrefillFragmentClaim<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.as_mut() {
            slot.state = TargetState::Failed;
        }
    }
}
/// Sequential scalar input following only the fragment's checked scatter map.
/// No raw offset, buffer export, refill, arbitrary callback or native authority.
#[derive(Debug)]
pub struct CapturePrefillFragmentWriter<'t, 'f, 'p, 'a> {
    slot: &'t mut TargetSlot,
    fragment: &'f CapturePrefillFragment<'p, 'a>,
    index: usize,
    cursor: usize,
    finished: bool,
}
impl CapturePrefillFragmentWriter<'_, '_, '_, '_> {
    /// Append one selected scalar to its unique geometry-derived destination.
    /// Failure poisons this target; its partial payload remains in the frame.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRunHostError> {
        let result = (|| {
            if self.slot.state == TargetState::Failed {
                return Err(CapturePrefillHostError::Incomplete { index: self.index }.into());
            }
            let tensor = self.slot.tensor.as_mut().expect("initialized target");
            tensor.custody.validate()?;
            let mapping = self
                .fragment
                .mapping_at(self.cursor)
                .ok_or(CapturePrefillHostError::Order)?;
            let next = tensor
                .covered
                .checked_add(1)
                .filter(|n| *n <= tensor.data.len())
                .ok_or(CapturePrefillHostError::Order)?;
            let destination = tensor
                .data
                .get_mut(mapping.destination_index())
                .ok_or(CapturePrefillHostError::Identity)?;
            *destination = value;
            tensor.covered = next;
            self.cursor += 1;
            Ok(())
        })();
        if result.is_err() {
            self.slot.state = TargetState::Failed;
        }
        result
    }
    /// Finish this exact fragment only. Coverage of all chunks and targets is
    /// checked separately; this cannot publish a partially initialized tensor.
    pub fn finish(mut self) -> Result<(), CaptureRunHostError> {
        self.slot
            .tensor
            .as_ref()
            .expect("initialized target")
            .custody
            .validate()?;
        if self.slot.state == TargetState::Failed || self.cursor != self.fragment.output_elements()
        {
            return Err(CapturePrefillHostError::Incomplete { index: self.index }.into());
        }
        self.slot.done = true;
        self.slot.state = TargetState::Active;
        self.finished = true;
        Ok(())
    }
}
impl Drop for CapturePrefillFragmentWriter<'_, '_, '_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            self.slot.state = TargetState::Failed;
        }
    }
}

mod assembled;
pub(in crate::working_memory::capture_run) use assembled::assembly_target_control_bytes;
