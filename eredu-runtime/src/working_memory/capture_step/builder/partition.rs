//! One prepaid evidence table owned by the same original frame through abort.
use super::*;
use crate::capture::partition::PreparedPartitionCaptureEvidence;
use eredu_nn::workspace::HostMetadataFunding;
use std::mem::{size_of, size_of_val};

impl PreparedCaptureStep<'_> {
    /// Prospective table construction and once-only publication controls for
    /// the exact admitted selection count. This query issues no frame or source.
    pub fn partition_evidence_metadata_bytes(rows: usize) -> Option<usize> {
        Self::partition_evidence_control_bytes(rows)?.checked_add(
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<PartitionCaptureEvidence>(
                rows,
            )?,
        )
    }
    fn partition_evidence_control_bytes(rows: usize) -> Option<usize> {
        let controls = [
            size_of::<(&mut Self, &HostMetadataFunding)>(),
            size_of::<CaptureStepError>(),
            size_of::<Result<(), CaptureStepError>>(),
            size_of::<Vec<PartitionCaptureEvidence>>() * 2,
            size_of::<PreparedPartitionCaptureEvidence>() * 2,
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<(&PartitionCaptureContext, &CaptureRecord)>(),
            size_of::<usize>() * 2,
            size_of::<(&CaptureTransform, &Option<CapturePayload>, bool)>(),
            size_of::<Option<&eredu_core::ObservationValueType>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)?
            .checked_mul(rows.checked_add(1)?)
    }
    pub(in crate::working_memory) fn prepare_partition_evidence(
        &mut self,
        metadata: &HostMetadataFunding,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        if self.partition_metadata.is_some() || !self.frame.partitions.is_empty() {
            return Err(CaptureStepError::PartitionState { index: 0 });
        }
        let bytes = Self::partition_evidence_control_bytes(self.plan.len())
            .ok_or(WorkingMemoryError::Overflow)?;
        metadata
            .reserve_metadata(bytes)
            .map_err(eredu_nn::workspace::WorkspaceMetadataError::from)
            .map_err(eredu_nn::Error::from)?;
        // Install custody before the first fallible destination constructor.
        // A failed preparation remains spent with this exact partial frame.
        self.partition_metadata = Some(metadata.clone());
        self.frame.partitions = metadata.metadata_vec(self.plan.len())?;
        Ok(())
    }

    pub(in crate::working_memory) fn record_partition_evidence(
        &mut self,
        evidence: PreparedPartitionCaptureEvidence,
    ) -> Result<(), CaptureStepError> {
        self.custody.validate()?;
        let context = &evidence.value.context;
        let index = context.selection_index;
        let invalid = || CaptureStepError::PartitionState { index };
        if !std::ptr::eq(self.plan.source, evidence.source.admission())
            || context.phase != self.plan.phase
            || context.prediction != self.plan.prediction
            || !context.matches_invocation(self.plan.invocation, self.plan.window)
            || !self
                .partition_metadata
                .as_ref()
                .is_some_and(|metadata| metadata.same_account(&evidence.metadata))
            || self.frame.partitions.len() == self.frame.partitions.capacity()
            || self
                .frame
                .partitions
                .iter()
                .any(|prior| prior.context.selection_index == index)
            || evidence.value.producers.is_empty()
        {
            return Err(invalid());
        }
        let record = self.frame.records.get(index).ok_or_else(invalid)?;
        let selection = self
            .plan
            .source
            .plan()
            .selections
            .get(index)
            .ok_or_else(invalid)?;
        let payload_matches = matches!(
            (&selection.transform, &record.payload),
            (CaptureTransform::Summary, Some(CapturePayload::Summary(_)))
                | (
                    CaptureTransform::Histogram { .. },
                    Some(CapturePayload::Histogram(_))
                )
                | (
                    CaptureTransform::TopCandidates { .. },
                    Some(CapturePayload::Candidates(_))
                )
                | (
                    CaptureTransform::TokenScores { .. },
                    Some(CapturePayload::TokenScores(_))
                )
                | (
                    CaptureTransform::FullTensor
                        | CaptureTransform::Slice
                        | CaptureTransform::Preview { .. },
                    Some(CapturePayload::SharedTensor(_))
                )
        ) || matches!(
            (
                &selection.transform,
                &record.payload,
                self.plan
                    .source
                    .points()
                    .get(index)
                    .map(|point| &point.value_type)
            ),
            (
                CaptureTransform::RoutedUnits,
                Some(CapturePayload::RoutedUnits(_)),
                Some(eredu_core::ObservationValueType::RoutedUnits { .. })
            )
        );
        if !matches!(
            record.outcome,
            CaptureOutcome::Captured | CaptureOutcome::Truncated { .. }
        ) || !payload_matches
            || record.charged != evidence.charged()
        {
            return Err(invalid());
        }
        // Complete, contiguous and routed receipts reach this seam through the same
        // closed evidence producer. Its final charge includes the actual global
        // assembly when present; no single contribution can stand in for it.
        // Already capacity checked; no allocation or metadata reconstruction.
        self.frame.partitions.push(evidence.value);
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::working_memory) fn partition_evidence(&self) -> &[PartitionCaptureEvidence] {
        &self.frame.partitions
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
