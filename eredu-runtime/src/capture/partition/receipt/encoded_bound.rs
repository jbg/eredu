//! Cold canonical envelopes bounded by the same admitted fragment records.
use super::*;
use eredu_core::capture::{BorrowedPartitionCaptureFragmentRecord, BorrowedPartitionCaptureProducerRecord};
use std::mem::{size_of, size_of_val};

impl PartitionCaptureReceiptPlan {
    pub(in crate::capture::partition) fn record_bound_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<BorrowedPartitionCaptureProducerRecord<'_, [(); 0]>>() * 2,
            size_of::<BorrowedPartitionCaptureFragmentRecord<'_, ()>>() * 2,
            crate::capture::RECORD_ENCODING_CONTROL_BYTES * 2,
            size_of::<CaptureUsage>() * 12, size_of::<sha2::Sha256>() * 3,
            size_of::<[u8; 32]>() * 3, size_of::<(u64, u64, u64, usize, usize, bool)>() * 3,
            size_of::<String>() * 2, size_of::<Result<String, CaptureError>>(),
            size_of::<Result<u64, CaptureError>>() * 3,
            size_of::<(&Self, usize, u64)>(),
        ];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(in crate::capture::partition) fn producer_envelope_bytes(&self, rank: usize) -> Result<u64, CaptureError> {
        let record = BorrowedPartitionCaptureProducerRecord {
            schema_version: PARTITION_CAPTURE_SCHEMA_VERSION, combination: self.combination,
            receipt_plan_identity: &self.identity, context: &self.context, producer_rank: rank,
            source_dtype: None, fragments: [(); 0],
        };
        crate::capture::encoded::encoded_size(&record).ok_or(CaptureError::Overflow)
    }
    pub(in crate::capture::partition) fn fragment_envelope_bytes(index: usize) -> Result<u64, CaptureError> {
        let fragment = BorrowedPartitionCaptureFragmentRecord { fragment_index: index, record: &() };
        let bytes = crate::capture::encoded::encoded_size(&fragment).ok_or(CaptureError::Overflow)?;
        // Remove only the canonical sentinel body; the full record is bounded
        // separately. The producer's empty [] already covers both delimiters.
        let null = crate::capture::encoded::encoded_size(&()).ok_or(CaptureError::Overflow)?;
        bytes.checked_sub(null).and_then(|n| n.checked_add(u64::from(index != 0)))
            .ok_or(CaptureError::Overflow)
    }
    pub(in crate::capture::partition) fn restrict_record_bytes(&mut self, bound: u64) -> Result<(), CaptureError> {
        if bound == 0 { return Err(CaptureError::Overflow); }
        let bound = self.limits.max_record_bytes.min(bound);
        if bound == self.limits.max_record_bytes { return Ok(()); }
        self.limits.max_record_bytes = bound;
        // Reuse the original paid SHA destination before this receipt is lent,
        // voted on or executed. No second String backing is allocated here.
        let mut identity = std::mem::take(&mut self.identity);
        identity.clear();
        self.identity = receipt_identity(&self.context, &self.producers, &self.funded,
            &self.routed, Some(identity), self.combination, self.world_size, self.limits, self.evidence_budget.as_ref())?;
        Ok(())
    }
}
