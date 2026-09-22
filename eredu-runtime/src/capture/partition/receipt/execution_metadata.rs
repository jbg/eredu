//! Prospective debits from the same admitted receipt and delivery workers.
use super::*;
use eredu_core::Completion;

impl PartitionCaptureReceiptPlan {
    /// Reconstruct this exact complete receipt, admit its local allowance, retain
    /// evidence, and encode or decode on the actual rank. Native transport,
    /// coordination and the enclosing program's entry frames are separate.
    pub fn complete_execution_metadata_bytes<T: PartitionCaptureTransport>(
        &self,
        local_rank: usize,
        run_length: usize,
    ) -> Option<usize>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.reconstruction_metadata_bytes(run_length)?
            .checked_add(PreparedPartitionCaptureAllowance::execution_metadata_bytes::<T>(self, local_rank)?)?
            .checked_add(PreparedPartitionCaptureEvidence::execution_metadata_bytes(self, run_length)?)?
            .checked_add(crate::working_memory::PreparedPartitionTensorDelivery::<T>::execution_metadata_bytes(self, local_rank)?)
    }

    /// Reconstruct this exact projected receipt, admit its fragment sources,
    /// retain evidence, and deliver the original destinations on the actual rank.
    /// The selected source wrapper, native callbacks and program entry frames
    /// remain separately quoted by their owning workers.
    pub fn fragment_execution_metadata_bytes<T: PartitionCaptureTransport>(
        &self,
        local_rank: usize,
        run_length: usize,
        prefill: bool,
    ) -> Option<usize>
    where
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        if local_rank >= self.world_size() {
            return None;
        }
        let fragments = self.producer(local_rank).map_or(0, |p| p.fragments().len());
        self.reconstruction_metadata_bytes(run_length)?
            .checked_add(PreparedPartitionFragmentAllowance::preparation_metadata_bytes::<T>(self)?)?
            .checked_add(PreparedPartitionFragmentAllowance::local_fragment_control_bytes()?.checked_mul(fragments)?)?
            .checked_add(PreparedPartitionCaptureEvidence::execution_metadata_bytes(self, run_length)?)?
            .checked_add(crate::working_memory::PreparedPartitionFragmentDelivery::<T>::execution_metadata_bytes(self, local_rank, prefill)?)
    }
}
