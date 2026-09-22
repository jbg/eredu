//! A scheduled global destination waiting for actual partition assembly.
use super::*;
use crate::capture::partition::PreparedPartitionAssemblyCharge;
use crate::working_memory::capture_run::claims::partition::{
    PartitionFragmentDelivered, PartitionFragmentDestination, PartitionFragmentValue,
};

impl<'a> ScheduledCaptureStep<'a> {
    /// Keep the same announced hook progression while local terms retain their
    /// own original host bank and native loans. This charge is issued only once
    /// by the actual globally funded fragment allowance, on producer and receiver.
    pub(crate) fn reserve_assembled_prefill_hook(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        charge: PreparedPartitionAssemblyCharge<'_>,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if !charge.validate(self.claim.source, index) {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let transform = &self
            .claim
            .source
            .admission()
            .plan()
            .selections
            .get(index)
            .ok_or(CapturePrefillHostError::Target { index })?
            .transform;
        if !matches!(
            transform,
            CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. }
                | CaptureTransform::Summary
                | CaptureTransform::Histogram { .. }
                | CaptureTransform::RoutedUnits
        ) {
            return Err(CapturePrefillHostError::Identity.into());
        }
        let previous = self
            .records()
            .get(index)
            .ok_or(CapturePrefillHostError::Target { index })?
            .charged;
        let total = charge.charged();
        let additional = CaptureUsage {
            captures: total
                .captures
                .checked_sub(previous.captures)
                .ok_or(CapturePrefillHostError::Identity)?,
            retained_bytes: total
                .retained_bytes
                .checked_sub(previous.retained_bytes)
                .ok_or(CapturePrefillHostError::Identity)?,
            host_bytes: total
                .host_bytes
                .checked_sub(previous.host_bytes)
                .ok_or(CapturePrefillHostError::Identity)?,
            encoded_bytes: total
                .encoded_bytes
                .checked_sub(previous.encoded_bytes)
                .ok_or(CapturePrefillHostError::Identity)?,
        };
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
            .and_then(|target| target.progression.as_mut())
            .ok_or(CapturePrefillHostError::Target { index })?;
        row.reserve_assembly_first(progress, &charge)
            .map_err(CapturePrefillHostError::from)?;
        let result = (|| {
            self.begin_prefill_target(index, dtype, additional)?;
            let targets = self.frame.prefill.as_mut().expect("validated prefill");
            let target = &mut targets.slots[index];
            if targets.next != 0
                || target.state != TargetState::Claimed
                || target.tensor.is_some()
                || target.completed.is_some()
                || target.summary.is_some()
                || target.histogram.is_some()
                || target.routed.is_some()
                || target.done
            {
                return Err(CapturePrefillHostError::Identity.into());
            }
            target.state = TargetState::Assembling;
            Ok(())
        })();
        if result.is_err() {
            self.fail_prefill_hook(index);
        }
        result
    }
    pub(crate) fn take_assembled_prefill_destination<'c>(
        &'c mut self,
        index: usize,
    ) -> Result<PartitionFragmentDestination<'a, 'c>, CaptureRunHostError> {
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
                .is_some_and(|target| target.state == TargetState::Assembling)
            || !self
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
        // Consume before a fallible host-plan constructor. The returned claim
        // uses this global frame's H, never a fragment's host identity.
        self.frame.prefill.as_mut().expect("checked prefill").slots[index].state =
            TargetState::AssemblyClaimed;
        let identity = ReceiptIdentity {
            phase: self.claim.phase,
            prediction: self.claim.prediction,
            index,
            custody: self.claim.custody.share_scheduled(),
        };
        let policy = self.claim.policy()?;
        Ok(
            match self.claim.source.admission().plan().selections[index].transform {
                CaptureTransform::RoutedUnits => {
                    PartitionFragmentDestination::Routed(CaptureRoutedClaim::from_partition_plan(
                        CaptureRoutedHostPlan::prepare(
                            policy
                                .routed_geometry(index)
                                .map_err(CaptureStepError::from)?,
                        )?,
                        identity,
                    ))
                }
                CaptureTransform::Summary => {
                    PartitionFragmentDestination::Summary(CaptureSummaryClaim::from_partition_plan(
                        CaptureSummaryHostPlan::prepare(
                            policy
                                .summary_geometry(index)
                                .map_err(CaptureStepError::from)?,
                        )?,
                        identity,
                    ))
                }
                CaptureTransform::Histogram { .. } => PartitionFragmentDestination::Histogram(
                    CaptureHistogramClaim::from_partition_plan(
                        CaptureHistogramHostPlan::prepare(
                            policy
                                .histogram_geometry(index)
                                .map_err(CaptureStepError::from)?,
                        )?,
                        identity,
                    ),
                ),
                CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. } => {
                    PartitionFragmentDestination::Tensor(CaptureTensorClaim {
                        plan: CaptureTensorHostPlan::prepare(
                            policy
                                .tensor_geometry(index)
                                .map_err(CaptureStepError::from)?,
                        )?,
                        identity,
                        exclusive: PhantomData,
                    })
                }
                _ => return Err(CapturePrefillHostError::Target { index }.into()),
            },
        )
    }
    /// Called only with the closed result released by the existing final
    /// all-rank delivery vote. Failed delivery never reaches this attachment.
    pub(crate) fn record_assembled_prefill(
        &mut self,
        value: PartitionFragmentDelivered,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let index = value.evidence.value.context.selection_index;
        let target = self
            .frame
            .prefill
            .as_ref()
            .and_then(|targets| targets.slots.get(index))
            .ok_or(CapturePrefillHostError::Target { index })?;
        if target.state != TargetState::AssemblyClaimed
            || target.dtype.as_ref() != Some(&value.dtype)
            || self.records().get(index).map(|record| record.charged) != Some(value.charged)
        {
            return Err(CapturePrefillHostError::Target { index }.into());
        }
        match value.value {
            PartitionFragmentValue::Routed(rows) => {
                self.record_assembled_routed_units(rows, value.dtype, CaptureUsage::default())?
            }
            PartitionFragmentValue::Tensor(tensor) => {
                self.record_tensor(tensor, value.dtype, CaptureUsage::default())?
            }
            PartitionFragmentValue::Summary(summary) => {
                self.record_summary(summary, value.dtype, CaptureUsage::default())?
            }
            PartitionFragmentValue::Histogram(histogram) => {
                self.record_histogram(histogram, value.dtype, CaptureUsage::default())?
            }
        }
        self.record_partition_evidence(value.evidence)?;
        self.frame
            .prefill
            .as_mut()
            .expect("validated prefill")
            .slots[index]
            .state = TargetState::AssemblyRecorded;
        Ok(())
    }
}
pub(in crate::working_memory::capture_run) fn assembly_target_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<PreparedPartitionAssemblyCharge<'_>>() * 2,
        size_of::<PartitionFragmentDelivered>() * 2,
        size_of::<PartitionFragmentDestination<'_, '_>>() * 2,
        size_of::<Result<PartitionFragmentDestination<'_, '_>, CaptureRunHostError>>(),
        size_of::<CaptureRunHostError>(),
        size_of::<CapturePrefillHostError>(),
        size_of::<Result<(), CaptureRunHostError>>(),
        size_of::<crate::capture::CapturePrefillObservationPolicy<'_>>(),
        size_of::<crate::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<(
            &mut ScheduledCaptureStep<'_>,
            usize,
            TensorDtype,
            PreparedPartitionAssemblyCharge<'_>,
        )>(),
        size_of::<(&mut ScheduledCaptureStep<'_>, PartitionFragmentDelivered)>(),
        size_of::<(&mut ScheduledCaptureStep<'_>, usize)>(),
        size_of::<CaptureUsage>() * 3,
        size_of::<ReceiptIdentity>(),
        size_of::<TensorDtype>() * 2,
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
