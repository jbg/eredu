//! Shared protocol debits composed from exact loaded rows and receipt sources.
use super::*;
use eredu_runtime::capture::partition::PreparedPartitionCaptureCoordination;

impl PartitionCaptureInvocationSource {
    /// This is the metadata consumed after binding the prepared program. The
    /// constructor query and native transport callback census remain separate.
    pub(in crate::composition::mlx) fn protocol_metadata_bytes<T: NativePartitionCaptureTransport>(
        &self,
        fragments: &[OwnedPartitionFragmentHostPlan],
        has_interventions: bool,
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        if self.source.admission().is_empty() && !has_interventions {
            return if fragments.is_empty() {
                Ok(0)
            } else {
                Err(memory(WorkingMemoryError::IdentityMismatch))
            };
        }
        let run_length = self.maximum_epoch_run_length().ok_or_else(overflow)?;
        let (artifact, execution, _) = self.loaded.source_labels();
        let mut bytes = 0usize;
        for part in [
            self.epoch_metadata_bytes(),
            // One prepare, one source/ledger coordination, one final delivery.
            PreparedPartitionCaptureProgram::<T>::entry_control_bytes()
                .and_then(|n| n.checked_mul(3)),
            eredu_runtime::working_memory::PreparedCaptureStep::partition_evidence_metadata_bytes(
                self.producers.len(),
            ),
            PreparedPartitionCaptureCoordination::<T>::prepare_metadata_bytes_for_labels(
                artifact.len(),
                execution.len(),
                run_length,
                self.overlay.as_ref().map(String::len),
                self.source.admission().identity().len(),
            ),
            PreparedPartitionCaptureCoordination::<T>::coordinate_metadata_bytes(),
        ] {
            bytes = bytes
                .checked_add(part.ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        bytes = bytes
            .checked_add(self.projected_execution_metadata_bytes::<T>()?)
            .ok_or_else(overflow)?;
        let included = PreparedPartitionCaptureCoordination::<T>::include_metadata_bytes()
            .ok_or_else(overflow)?;
        let skipped = PreparedPartitionCaptureCoordination::<T>::skip_metadata_bytes()
            .ok_or_else(overflow)?;
        let mut projected = 0usize;
        let mut complete = 0usize;
        for &(index, _) in &self.protocol {
            let mode = self
                .protocol_row_mode(index)
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
            let (receipt, execution, vote, entry) = match mode {
                PartitionCaptureRowMode::Complete => {
                    let receipt = self
                        .complete_receipts
                        .iter()
                        .find(|r| r.context().selection_index == index)
                        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
                    complete = complete.checked_add(1).ok_or_else(overflow)?;
                    (
                        receipt,
                        receipt.complete_execution_metadata_bytes::<T>(self.rank, run_length),
                        PreparedPartitionCaptureCoordination::<T>::selected_source_metadata_bytes(),
                        PreparedPartitionCaptureProgram::<T>::terminal_row_control_bytes(),
                    )
                }
                PartitionCaptureRowMode::Contiguous | PartitionCaptureRowMode::Routed => {
                    let host = fragments
                        .iter()
                        .find(|p| p.receipt().context().selection_index == index)
                        .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?;
                    if !host.matches_invocation(
                        &self.source,
                        self.invocation.phase,
                        self.invocation.prediction,
                        self.invocation.physical,
                        self.invocation.window,
                    ) {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    let receipt = host.receipt();
                    projected = projected.checked_add(1).ok_or_else(overflow)?;
                    let vote = match mode {
                        PartitionCaptureRowMode::Contiguous => PreparedPartitionCaptureCoordination::<
                            T,
                        >::contiguous_source_metadata_bytes(
                        ),
                        PartitionCaptureRowMode::Routed => {
                            PreparedPartitionCaptureCoordination::<T>::routed_source_metadata_bytes(
                                receipt,
                            )
                        }
                        _ => unreachable!(),
                    };
                    (
                        receipt,
                        receipt
                            .fragment_execution_metadata_bytes::<T>(self.rank, run_length, false),
                        vote,
                        // The same row is prepared, coordinated and delivered once.
                        PreparedPartitionCaptureProgram::<T>::projected_entry_control_bytes()
                            .and_then(|n| n.checked_mul(3)),
                    )
                }
            };
            if !receipt.shared_plan_source().same_storage(&self.source) {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let vote = vote.ok_or_else(overflow)?.max(
                PreparedPartitionCaptureCoordination::<T>::selected_source_metadata_bytes()
                    .ok_or_else(overflow)?,
            );
            for part in [execution, Some(vote), entry, Some(included.max(skipped))] {
                bytes = bytes
                    .checked_add(part.ok_or_else(overflow)?)
                    .ok_or_else(overflow)?;
            }
        }
        if complete != self.complete_receipts.len() || projected != fragments.len() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(bytes)
    }
    /// The actual cold traversal identifies local callbacks. Each source can
    /// follow either its producing or receiving branch, with no repeated hook.
    pub(in crate::composition::mlx) fn protocol_hook_metadata_bytes<
        T: NativePartitionCaptureTransport,
    >(
        &self,
        scalars: &[Option<WorkspaceFloatingType>],
    ) -> Result<usize, Error>
    where
        <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
    {
        if scalars.len() != self.producers.len() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let hook = PreparedPartitionCaptureProgram::<T>::invocation_hook_control_bytes()
            .ok_or_else(overflow)?;
        self.protocol
            .iter()
            .filter(|(index, _)| scalars[*index].is_some())
            .try_fold(0usize, |bytes, _| {
                bytes.checked_add(hook).ok_or_else(overflow)
            })
    }
}
