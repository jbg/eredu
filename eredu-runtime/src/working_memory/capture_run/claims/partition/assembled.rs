//! A once-only global target for the existing ordinary fragment assembler.
use super::fragments::{
    PartitionFragmentDelivered, PartitionFragmentDestination, PartitionFragmentValue,
};
use super::*;
use crate::capture::partition::PreparedPartitionAssemblyCharge;

/// Issued alongside the original destination. The closed delivery worker must
/// return the agreed value to this same source, coordinate, scalar and account.
pub(in crate::working_memory::capture_run) struct InvocationAssemblyTarget<'a> {
    source: &'a SharedCapturePlan,
    identity: ReceiptIdentity,
    dtype: TensorDtype,
    charged: CaptureUsage,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
}
impl<'a> ScheduledCaptureStep<'a> {
    pub(in crate::working_memory::capture_run) fn take_assembled_invocation_destination<'c>(
        &'c mut self,
        index: usize,
        dtype: TensorDtype,
        charge: PreparedPartitionAssemblyCharge<'_>,
    ) -> Result<
        (
            PartitionFragmentDestination<'a, 'c>,
            InvocationAssemblyTarget<'a>,
        ),
        CaptureRunHostError,
    > {
        self.claim.custody.validate()?;
        if self.frame.prefill.is_some()
            || !charge.validate_invocation(
                self.claim.source,
                index,
                self.claim.phase,
                self.claim.prediction,
                self.claim.invocation,
                self.claim.window,
            )
            || !self
                .records()
                .get(index)
                .is_some_and(|record| matches!(record.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let target = InvocationAssemblyTarget {
            source: self.claim.source,
            identity: ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            dtype,
            charged: charge.charged(),
            invocation: self.claim.invocation,
            window: self.claim.window,
        };
        let destination = match &self.claim.source.admission().plan().selections[index].transform {
            CaptureTransform::RoutedUnits => {
                PartitionFragmentDestination::Routed(self.take_routed_units(index)?)
            }
            CaptureTransform::Summary => {
                PartitionFragmentDestination::Summary(self.take_summary(index)?)
            }
            CaptureTransform::Histogram { .. } => {
                PartitionFragmentDestination::Histogram(self.take_histogram(index)?)
            }
            CaptureTransform::FullTensor
            | CaptureTransform::Slice
            | CaptureTransform::Preview { .. } => {
                PartitionFragmentDestination::Tensor(self.take_tensor(index)?)
            }
            _ => return Err(CaptureRunHostError::ReceiptMismatch),
        };
        Ok((destination, target))
    }
}
impl InvocationAssemblyTarget<'_> {
    /// Only the existing final delivery vote releases this closed result.
    /// Its already-reserved global record charge is attached once; no ledger
    /// is charged again and no native allowance is recreated here.
    pub(in crate::working_memory::capture_run) fn record(
        self,
        frame: &mut ScheduledCaptureStep<'_>,
        value: PartitionFragmentDelivered,
    ) -> Result<(), CaptureRunHostError> {
        self.identity.custody.validate()?;
        let index = self.identity.index;
        let context = &value.evidence.value.context;
        if !self.source.same_storage(frame.claim.source)
            || !self.identity.custody.same_schedule(&frame.claim.custody)
            || self.identity.phase != frame.claim.phase
            || self.identity.prediction != frame.claim.prediction
            || frame.frame.prefill.is_some()
            || frame.claim.window != self.window
            || frame.claim.invocation != self.invocation
            || !context.matches_invocation(self.invocation, self.window)
            || value.dtype != self.dtype
            || value.charged != self.charged
            || context.selection_index != index
            || context.phase != self.identity.phase
            || context.prediction != self.identity.prediction
            || context.capture_plan_identity != self.source.admission().identity()
            || !frame
                .records()
                .get(index)
                .is_some_and(|record| matches!(record.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let previous = frame.records()[index].charged;
        let additional = CaptureUsage {
            captures: self
                .charged
                .captures
                .checked_sub(previous.captures)
                .ok_or(CaptureRunHostError::ReceiptMismatch)?,
            retained_bytes: self
                .charged
                .retained_bytes
                .checked_sub(previous.retained_bytes)
                .ok_or(CaptureRunHostError::ReceiptMismatch)?,
            host_bytes: self
                .charged
                .host_bytes
                .checked_sub(previous.host_bytes)
                .ok_or(CaptureRunHostError::ReceiptMismatch)?,
            encoded_bytes: self
                .charged
                .encoded_bytes
                .checked_sub(previous.encoded_bytes)
                .ok_or(CaptureRunHostError::ReceiptMismatch)?,
        };
        match value.value {
            PartitionFragmentValue::Routed(rows) => {
                frame.record_assembled_routed_units(rows, value.dtype, additional)?
            }
            PartitionFragmentValue::Tensor(tensor) => {
                frame.record_tensor(tensor, value.dtype, additional)?
            }
            PartitionFragmentValue::Summary(summary) => {
                frame.record_summary(summary, value.dtype, additional)?
            }
            PartitionFragmentValue::Histogram(histogram) => {
                frame.record_histogram(histogram, value.dtype, additional)?
            }
        }
        frame.record_partition_evidence(value.evidence)
    }
}
pub(in crate::working_memory::capture_run) fn invocation_assembly_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<InvocationAssemblyTarget<'_>>() * 2,
        size_of::<PreparedPartitionAssemblyCharge<'_>>(),
        size_of::<PartitionFragmentDestination<'_, '_>>(),
        size_of::<PartitionFragmentDelivered>(),
        size_of::<(
            PartitionFragmentDestination<'_, '_>,
            InvocationAssemblyTarget<'_>,
        )>(),
        size_of::<
            Result<
                (
                    PartitionFragmentDestination<'_, '_>,
                    InvocationAssemblyTarget<'_>,
                ),
                CaptureRunHostError,
            >,
        >(),
        size_of::<(
            &mut ScheduledCaptureStep<'_>,
            usize,
            TensorDtype,
            PreparedPartitionAssemblyCharge<'_>,
        )>(),
        size_of::<(
            InvocationAssemblyTarget<'_>,
            &mut ScheduledCaptureStep<'_>,
            PartitionFragmentDelivered,
        )>(),
        size_of::<CaptureUsage>() * 3,
        size_of::<ReceiptIdentity>(),
        size_of::<TensorDtype>(),
        size_of::<CaptureRunHostError>(),
        size_of::<Result<(), CaptureRunHostError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
