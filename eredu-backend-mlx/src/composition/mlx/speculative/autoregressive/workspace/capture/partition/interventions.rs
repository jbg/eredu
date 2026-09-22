//! Sealed model member/evidence sources composed into the existing role quote.
use super::*;
use eredu_runtime::capture::partition::{
    PreparedPartitionCaptureCoordination as Coordination,
    PreparedPartitionCaptureProgram as Program, PreparedPartitionInterventionEvidence as Evidence,
    PreparedPartitionInterventionSource as Source,
};
impl PartitionQuote {
    pub(super) fn prepare_interventions(
        &mut self,
        native: &OriginalModelPartitionSource,
        edits: &PreparedModelInterventions,
        context: &WorkspaceContext,
    ) -> Result<Vec<(usize, [OwnedPartitionFragmentHostPlan; 2])>, Error> {
        let count = edits.source().plan().admission().plan().operations.len();
        let funding = context
            .metadata_funding()
            .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        context
            .charge_metadata(
                size_of::<(
                    &mut Self,
                    &OriginalModelPartitionSource,
                    &PreparedModelInterventions,
                    &WorkspaceContext,
                )>()
                .checked_add(size_of::<Vec<(usize, [OwnedPartitionFragmentHostPlan; 2])>>())
                .ok_or_else(overflow)?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let source = edits
            .partition_sources_metadata_bytes(context)
            .map_err(Error::Neural)?;
        self.add_metadata(source)?;
        self.add_metadata(
            Program::<Transport>::intervention_sources_metadata_bytes::<
                std::vec::IntoIter<Option<Source>>,
                std::vec::IntoIter<Option<Evidence<'_, Transport>>>,
            >(count)
            .ok_or_else(overflow)?,
        )?;
        for index in 0..count {
            if !edits
                .partition_operation_selected(index)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            {
                continue;
            }
            self.add_metadata(
                Source::execution_control_bytes::<Transport>().ok_or_else(overflow)?,
            )?;
            self.add_metadata(
                Coordination::<Transport>::intervention_metadata_bytes().ok_or_else(overflow)?,
            )?;
            for demand in Source::world_transport_demands() {
                Self::add(&mut self.capacity, &mut self.metadata, demand, native)?;
            }
            let members = edits
                .partition_members(index)
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let mut ranks = context
                .metadata_vec(members.len())
                .map_err(|cause| Error::Neural(cause.into()))?;
            ranks.extend(members.map(|member| member.rank));
            if ranks.len() > 1 && ranks.contains(&native.rank()) {
                let quote = native.control().capture_member_vote_requirements(&ranks)?;
                for _ in 0..2 {
                    self.capacity.graph = self
                        .capacity
                        .graph
                        .checked_add(quote.capacity.graph)
                        .ok_or_else(overflow)?;
                    self.capacity.records = self
                        .capacity
                        .records
                        .checked_add(quote.capacity.records)
                        .ok_or_else(overflow)?;
                    self.capacity.backing = self
                        .capacity
                        .backing
                        .checked_add(quote.capacity.backing)
                        .ok_or_else(overflow)?;
                    self.add_metadata(quote.metadata)?;
                }
            }
        }
        let (evidence, plans) =
            native.prepare_evidence(self.source.capture_source(), edits, &funding)?;
        self.add_metadata(evidence.execution_metadata_bytes::<Transport>()?)?;
        self.add_metadata(
            partition::PartitionCaptureFrame::<Transport>::model_evidence_attachment_control_bytes(
            )
            .ok_or_else(overflow)?,
        )?;
        let run_length = evidence.maximum_epoch_run_length().ok_or_else(overflow)?;
        let operations = evidence.operation_indices().count();
        if operations != 0 {
            // Shared parent preparation and final delivery each enter evidence once.
            self.add_metadata(
                Program::<Transport>::evidence_frame_control_bytes()
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(overflow)?,
            )?;
        }
        for (operation, side, receipt, scalar) in evidence.receipts(&plans)? {
            let coord = receipt.context();
            if side == 0 {
                let parts = [
                    Evidence::<Transport>::construction_control_bytes(),
                    Evidence::<Transport>::budget_binding_control_bytes(),
                    evidence.epoch_metadata_bytes(),
                    Program::<Transport>::entry_control_bytes().and_then(|n|n.checked_mul(3)),
                    eredu_runtime::working_memory::PreparedCaptureStep::partition_evidence_metadata_bytes(2),
                    Coordination::<Transport>::prepare_metadata_bytes_with_run_length(coord,run_length),
                    Coordination::<Transport>::coordinate_metadata_bytes(),
                ];
                for part in parts {
                    self.add_metadata(part.ok_or_else(overflow)?)?;
                }
                for demand in
                    eredu_runtime::capture::partition::partition_capture_coordination_demands(
                        receipt.shared_plan_source(),
                        coord.phase,
                        coord.prediction,
                    )
                {
                    Self::add(&mut self.capacity, &mut self.metadata, demand, native)?;
                }
            }
            for part in [
                receipt.fragment_execution_metadata_bytes::<Transport>(
                    native.rank(),
                    run_length,
                    false,
                ),
                Some(evidence.projected_execution_metadata_bytes::<Transport>(operation, side)?),
                Program::<Transport>::projected_entry_control_bytes()
                    .and_then(|n| n.checked_mul(3)),
                Some(
                    Coordination::<Transport>::contiguous_source_metadata_bytes()
                        .ok_or_else(overflow)?
                        .max(
                            Coordination::<Transport>::selected_source_metadata_bytes()
                                .ok_or_else(overflow)?,
                        ),
                ),
                Some(
                    Coordination::<Transport>::include_metadata_bytes()
                        .ok_or_else(overflow)?
                        .max(
                            Coordination::<Transport>::skip_metadata_bytes()
                                .ok_or_else(overflow)?,
                        ),
                ),
                partition::native::model_control_bytes(),
            ] {
                self.add_metadata(part.ok_or_else(overflow)?)?;
            }
            if scalar.is_some() {
                self.add_metadata(
                    Program::<Transport>::invocation_hook_control_bytes().ok_or_else(overflow)?,
                )?;
            }
            for demand in receipt
                .transport_demands()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?
            {
                Self::add(&mut self.capacity, &mut self.metadata, demand, native)?;
            }
        }
        self.evidence = Some(evidence);
        Ok(plans)
    }
    fn add_metadata(&mut self, bytes: usize) -> Result<(), Error> {
        self.metadata = self.metadata.checked_add(bytes).ok_or_else(overflow)?;
        Ok(())
    }
}
