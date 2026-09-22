//! The retained remote producer fills one original target after prompt progress.
use super::*;

impl<'a> ScheduledCaptureStep<'a> {
    pub(crate) fn record_partition_skip(
        &mut self,
        index: usize,
        reason: CaptureSkipReason,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if !matches!(reason, CaptureSkipReason::Limit { .. }) {
            return Err(CapturePrefillHostError::Identity.into());
        }
        if let Some(targets) = self.frame.prefill.as_mut() {
            let policy = crate::capture::CapturePrefillObservationPolicy::new(
                self.claim.source,
                targets.inference,
            )
            .map_err(CapturePrefillHostError::from)?;
            let row = policy.row(index).map_err(CapturePrefillHostError::from)?;
            let progress = targets
                .slots
                .get_mut(index)
                .and_then(|slot| slot.progression.as_mut())
                .ok_or(CapturePrefillHostError::Identity)?;
            row.skip_partition_before_hook(progress)
                .map_err(CapturePrefillHostError::from)?;
        }
        self.record_skip(index, reason, None, CaptureUsage::default())
    }

    /// A receiver advances the same scheduled hook and spends its existing H
    /// target, carrying the remote producer's already admitted logical charge.
    pub(crate) fn reserve_remote_prefill_hook(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        charge: crate::capture::partition::PreparedPartitionRemoteCharge<'_>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let policy = crate::capture::CapturePrefillObservationPolicy::new(
            self.claim.source,
            targets.inference,
        )
        .map_err(CapturePrefillHostError::from)?;
        let row = policy.row(index).map_err(CapturePrefillHostError::from)?;
        let progress = targets
            .slots
            .get_mut(index)
            .and_then(|slot| slot.progression.as_mut())
            .ok_or(CapturePrefillHostError::Identity)?;
        row.reserve_remote_first(progress, &charge)
            .map_err(CapturePrefillHostError::from)?;
        let result = self
            .begin_prefill_target(index, dtype, charge.usage())
            .and_then(|()| self.mark_remote_prefill_target(index));
        if result.is_err() {
            self.fail_prefill_hook(index);
        }
        result
    }

    /// Mark an already reserved tensor target as remotely produced. This only
    /// selects host destination behavior. The enclosing partition observer must
    /// supply the exact retained placement and all-rank transport authority.
    pub(crate) fn mark_remote_prefill_target(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let policy = self.claim.policy()?;
        let coordinates = if matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::Summary
        ) {
            let geometry = policy
                .summary_geometry(index)
                .map_err(CaptureStepError::from)?;
            (geometry.phase(), geometry.prediction())
        } else if matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::Histogram { .. }
        ) {
            let geometry = policy
                .histogram_geometry(index)
                .map_err(CaptureStepError::from)?;
            (geometry.phase(), geometry.prediction())
        } else if matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::TopCandidates { .. }
        ) {
            let geometry = CaptureCandidateGeometry::prepare(
                self.claim.source.admission(),
                index,
                self.claim.phase,
                self.claim.prediction,
                self.claim.invocation,
            )
            .map_err(CaptureStepError::from)?;
            (geometry.phase(), geometry.prediction())
        } else if matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::TokenScores { .. }
        ) {
            let geometry = CaptureTokenScoreGeometry::prepare(
                self.claim.source.admission(),
                index,
                self.claim.phase,
                self.claim.prediction,
                self.claim.invocation,
            )
            .map_err(CaptureStepError::from)?;
            (geometry.phase(), geometry.prediction())
        } else {
            let geometry = policy
                .tensor_geometry(index)
                .map_err(CaptureStepError::from)?;
            (geometry.phase(), geometry.prediction())
        };
        if coordinates != (CapturePhase::Prefill, 0) {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let targets = self
            .frame
            .prefill
            .as_mut()
            .ok_or(CapturePrefillHostError::Identity)?;
        let target = targets
            .slots
            .get_mut(index)
            .ok_or(CapturePrefillHostError::Target { index })?;
        let terminal = matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        );
        let expected_chunk = if terminal {
            targets
                .inference
                .input_positions
                .div_ceil(targets.inference.prefill_chunk_positions)
                - 1
        } else {
            0
        };
        if targets.next != expected_chunk
            || target.state != TargetState::Claimed
            || target.tensor.is_some()
            || target.completed.is_some()
            || target.summary.is_some()
            || target.histogram.is_some()
            || target.progression.is_none()
            || target.done
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Spent)
        {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        target.state = TargetState::Remote;
        Ok(())
    }

    /// After every announced chunk completes, consume the single remote target
    /// claim. Reuses the ordinary direct receipt decoder and original H; no
    /// fragment/offset, source replacement, native grant or reset is exposed.
    pub(crate) fn take_remote_prefill_tensor<'c>(
        &'c mut self,
        index: usize,
    ) -> Result<CaptureTensorClaim<'a, 'c>, CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        let geometry = self
            .claim
            .policy()?
            .tensor_geometry(index)
            .map_err(CaptureStepError::from)?;
        let plan = CaptureTensorHostPlan::prepare(geometry)?;
        self.frame.prefill.as_mut().expect("checked targets").slots[index].state =
            TargetState::RemoteClaimed;
        Ok(CaptureTensorClaim {
            plan,
            identity: ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            exclusive: PhantomData,
        })
    }

    pub(crate) fn validate_remote_prefill_claim(
        &self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
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
            || !targets
                .slots
                .get(index)
                .is_some_and(|target| target.state == TargetState::Remote)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|record| matches!(record.outcome, CaptureOutcome::Missing))
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Spent)
        {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        Ok(())
    }
    pub(crate) fn begin_remote_prefill_summary_claim(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        if !matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::Summary
        ) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        self.frame.prefill.as_mut().expect("checked targets").slots[index].state =
            TargetState::RemoteClaimed;
        Ok(())
    }
    pub(crate) fn begin_remote_prefill_histogram_claim(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        if !matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::Histogram { .. }
        ) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        self.frame.prefill.as_mut().expect("checked targets").slots[index].state =
            TargetState::RemoteClaimed;
        Ok(())
    }
    pub(crate) fn validate_remote_prefill_record(
        &self,
        index: usize,
        source_dtype: &TensorDtype,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let target = self
            .frame
            .prefill
            .as_ref()
            .and_then(|targets| targets.slots.get(index))
            .ok_or(CapturePrefillHostError::Target { index })?;
        if target.state != TargetState::RemoteClaimed || target.dtype.as_ref() != Some(source_dtype)
        {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        Ok(())
    }
    pub(crate) fn mark_remote_prefill_recorded(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let target = self
            .frame
            .prefill
            .as_mut()
            .and_then(|targets| targets.slots.get_mut(index))
            .ok_or(CapturePrefillHostError::Target { index })?;
        if target.state != TargetState::RemoteClaimed {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        target.state = TargetState::RemoteRecorded;
        Ok(())
    }

    /// Attach the decoded receipt to its exact spent remote target. The original
    /// reservation already contains the full record charge; receipt attachment
    /// spends nothing twice and proves no all-rank delivery or native settlement.
    pub(crate) fn record_remote_prefill_tensor(
        &mut self,
        tensor: ClaimedCaptureTensor,
        source_dtype: TensorDtype,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let index = tensor.identity.index;
        self.validate_remote_prefill_record(index, &source_dtype)?;
        self.record_tensor(tensor, source_dtype, CaptureUsage::default())?;
        self.mark_remote_prefill_recorded(index)
    }
}

impl ScheduledCaptureStep<'_> {
    /// The retained schedule alone selects the final physical source extent.
    pub(crate) fn partition_terminal_rows(
        &self,
        index: usize,
    ) -> Result<Option<usize>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        let selection = self
            .claim
            .source
            .admission()
            .plan()
            .selections
            .get(index)
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?;
        if !matches!(
            selection.transform,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        ) {
            return Ok(None);
        }
        let Some(targets) = self.frame.prefill.as_ref() else {
            return Ok(None);
        };
        let geometry = targets.inference;
        let rows = if geometry.output == eredu_core::OutputDemand::Sequence {
            geometry
                .input_positions
                .checked_sub(1)
                .and_then(|n| n.checked_rem(geometry.prefill_chunk_positions))
                .and_then(|n| n.checked_add(1))
                .ok_or(WorkingMemoryError::Overflow)?
        } else {
            1
        };
        Ok(Some(
            usize::try_from(rows).map_err(|_| WorkingMemoryError::Overflow)?,
        ))
    }
    pub(crate) fn begin_remote_terminal_claim(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.validate_remote_prefill_claim(index)?;
        if !matches!(
            self.claim.source.admission().plan().selections[index].transform,
            CaptureTransform::TopCandidates { .. } | CaptureTransform::TokenScores { .. }
        ) {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        self.frame.prefill.as_mut().expect("checked targets").slots[index].state =
            TargetState::RemoteClaimed;
        Ok(())
    }
    pub(crate) fn finish_remote_terminal_record(
        &mut self,
        index: usize,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let target = self
            .frame
            .prefill
            .as_mut()
            .and_then(|targets| targets.slots.get_mut(index))
            .ok_or(CapturePrefillHostError::Target { index })?;
        if target.state != TargetState::Recorded {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        target.state = TargetState::RemoteRecorded;
        Ok(())
    }
}
