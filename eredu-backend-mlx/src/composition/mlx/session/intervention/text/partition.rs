//! Partition source attachment preserves the existing ordinary row schedule.
use super::*;
use eredu_architectures::component_partition::ComponentPartitionLayouts;
use eredu_runtime::RetainedCommunicationSource;
pub(super) struct Binding {
    pub(super) rank: usize,
    pub(super) world: usize,
    pub(super) communication: RetainedCommunicationSource,
}
impl PreparedTextInterventions {
    pub(crate) fn bind_partition(
        &mut self,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        communication: &RetainedCommunicationSource,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if self.active.is_some()
            || self.prefill_active.is_some()
            || self.next_prediction != self.first_prediction
            || self.partition.is_some()
        {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        for row in &mut self.rows {
            row.bind_partition(layouts, rank, communication, context)?;
        }
        self.partition = Some(Binding {
            rank,
            world: layouts.topology().world_size(),
            communication: communication.clone(),
        });
        Ok(())
    }
    pub(crate) fn validate_partition(
        &self,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        communication: &RetainedCommunicationSource,
    ) -> std::result::Result<(), Failure> {
        let binding = self.partition.as_ref().ok_or(Failure::SourceChanged)?;
        if binding.rank != rank
            || binding.world != layouts.topology().world_size()
            || !binding.communication.same_source(communication)
        {
            return Err(Failure::SourceChanged);
        }
        Ok(())
    }
    pub(crate) fn begin_partition_prefill(
        &mut self,
        inference: eredu_core::InferenceGeometry,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        layouts: &ComponentPartitionLayouts,
        rank: usize,
        communication: &RetainedCommunicationSource,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.validate_partition(layouts, rank, communication)
            .map_err(|cause| context.metadata_source(cause))?;
        self.begin_prefill_with_partition(
            inference,
            chunk,
            Some((layouts, rank, communication)),
            context,
        )
    }
    /// Actual local side representation, preserved across the original chunk
    /// sequence. Remote descriptors cannot supply a floating witness.
    pub(crate) fn partition_evidence_scalar(&self,phase:CapturePhase,prediction:u64,
        operation:usize,side:eredu_core::capture::InterventionEvidenceSide,
        metadata:&HostMetadataFunding)
        ->std::result::Result<Option<eredu_nn::workspace::WorkspaceFloatingType>,eredu_nn::Error> {
        use eredu_nn::workspace::WorkspaceFloatingType;
        metadata.reserve_metadata(size_of::<(&Self,CapturePhase,u64,usize,
            eredu_core::capture::InterventionEvidenceSide,&HostMetadataFunding)>()
            +size_of::<(Option<bool>,Option<WorkspaceFloatingType>,Option<WorkspaceFloatingType>,usize)>()
            +size_of::<std::result::Result<Option<WorkspaceFloatingType>,eredu_nn::Error>>()
            +size_of::<std::result::Result<Option<&super::super::PreparedPartitionModelIntervention>,Failure>>()
            +size_of::<&super::super::partition::PreparedPartitionEvidenceSource>())
            .map_err(|cause|metadata.metadata_source(cause))?;
        let binding=self.partition.as_ref().ok_or_else(||metadata.metadata_source(Failure::SourceChanged))?;
        let terminal=self.row(phase,prediction).map_err(|cause|metadata.metadata_source(cause))?;
        let plan=self.source.plan().admission();
        let point=plan.points().get(operation).ok_or_else(||metadata.metadata_source(Failure::SourceChanged))?;
        let companion=self.source.plan().evidence(operation).ok_or_else(||metadata.metadata_source(Failure::SourceChanged))?;
        let windowed=phase==CapturePhase::Prefill&&eredu_runtime::intervention::InterventionPrefillWindow::row_axis(point);
        let count=if windowed{self.prefill_rows.len()}else{1};
        if count==0{return Err(metadata.metadata_source(Failure::SourceChanged));}
        let mut present=None;let mut scalar=None;
        for ordinal in 0..count {
            let row=if windowed{&self.prefill_rows[ordinal]}else{terminal};
            let local=row.partition_source(operation).map_err(|cause|metadata.metadata_source(cause))?;
            if present.is_some_and(|expected|expected!=local.is_some()) {return Err(metadata.metadata_source(Failure::SourceChanged));}
            present=Some(local.is_some());
            let Some(local)=local else{continue};
            let source=local.evidence().ok_or_else(||metadata.metadata_source(Failure::SourceChanged))?;
            if source.rank()!=binding.rank||!source.source().same_storage(companion.shared_geometry_source()) {
                return Err(metadata.metadata_source(Failure::SourceChanged));
            }
            let actual=source.scalar(side).ok_or_else(||metadata.metadata_source(Failure::SourceChanged))?;
            if scalar.is_some_and(|expected|expected!=actual){return Err(metadata.metadata_source(Failure::ShapeMismatch));}
            scalar=Some(actual);
        }
        Ok(scalar)
    }
    /// Copy the exact sealed per-member transcript into this frame's original
    /// metadata account. No peer tensor is built and no quota is issued here.
    pub(crate) fn partition_sources(
        &self,
        phase: CapturePhase,
        prediction: u64,
        metadata: &HostMetadataFunding,
    ) -> std::result::Result<
        Vec<Option<eredu_runtime::capture::partition::PreparedPartitionInterventionSource>>,
        eredu_nn::Error,
    > {
        use eredu_runtime::capture::partition::{
            PartitionInterventionInvocationSource, PartitionInterventionMemberSource,
            PreparedPartitionInterventionSource,
        };
        metadata
            .reserve_metadata(source_control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)
            .map_err(|cause| metadata.metadata_source(cause))?;
        let binding = self
            .partition
            .as_ref()
            .ok_or_else(|| metadata.metadata_source(Failure::SourceChanged))?;
        let terminal = self
            .row(phase, prediction)
            .map_err(|cause| metadata.metadata_source(cause))?;
        let plan = self.source.plan().admission();
        let mut output = metadata.metadata_vec(plan.plan().operations.len())?;
        for (index, operation) in plan.plan().operations.iter().enumerate() {
            if !operation.schedule.includes(phase, prediction) {
                output.push(None);
                continue;
            }
            let windowed = phase == CapturePhase::Prefill
                && eredu_runtime::intervention::InterventionPrefillWindow::row_axis(
                    &plan.points()[index],
                );
            let count = if windowed { self.prefill_rows.len() } else { 1 };
            if count == 0 {
                return Err(metadata.metadata_source(Failure::SourceChanged));
            }
            let mut members = metadata.metadata_vec(count)?;
            for ordinal in 0..count {
                let row = if windowed {
                    &self.prefill_rows[ordinal]
                } else {
                    terminal
                };
                let source = row
                    .partition_members(index)
                    .map_err(|cause| metadata.metadata_source(cause))?;
                let mut copied = metadata.metadata_vec(source.len())?;
                copied.extend(source);
                members.push(copied);
            }
            let mut invocations = metadata.metadata_vec(count)?;
            for (ordinal, members) in members.iter().enumerate() {
                invocations.push(PartitionInterventionInvocationSource {
                    window: if windowed {
                        self.prefill_rows[ordinal].scheduled_span()
                    } else {
                        None
                    },
                    members,
                });
            }
            output.push(Some(
                PreparedPartitionInterventionSource::new(
                    &self.source,
                    index,
                    phase,
                    prediction,
                    binding.world,
                    &invocations,
                    metadata,
                )
                .map_err(|cause| metadata.metadata_source(cause))?,
            ));
        }
        Ok(output)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Binding>(),
        size_of::<Option<Binding>>(),
        size_of::<RetainedCommunicationSource>(),
        size_of::<(
            &mut PreparedTextInterventions,
            &ComponentPartitionLayouts,
            usize,
            &RetainedCommunicationSource,
            &WorkspaceContext,
        )>(),
        size_of::<(
            &mut PreparedTextInterventions,
            eredu_core::InferenceGeometry,
            &eredu_runtime::prefill::PrefillChunk,
            &ComponentPartitionLayouts,
            usize,
            &RetainedCommunicationSource,
            &WorkspaceContext,
        )>(),
        size_of::<Result<()>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

fn source_control_bytes() -> Option<usize> {
    use eredu_runtime::capture::partition::{
        PartitionInterventionInvocationSource, PartitionInterventionMemberSource,
        PreparedPartitionInterventionSource,
    };
    let frames = [
        control_bytes()?,
        size_of::<Vec<Vec<PartitionInterventionMemberSource<'_>>>>(),
        size_of::<Vec<PartitionInterventionMemberSource<'_>>>(),
        size_of::<PartitionInterventionMemberSource<'_>>(),
        size_of::<Vec<PartitionInterventionInvocationSource<'_>>>(),
        size_of::<PartitionInterventionInvocationSource<'_>>(),
        size_of::<Vec<Option<PreparedPartitionInterventionSource>>>(),
        size_of::<Option<PreparedPartitionInterventionSource>>(),
        size_of::<(
            &PreparedTextInterventions,
            CapturePhase,
            u64,
            &HostMetadataFunding,
        )>(),
        size_of::<
            std::result::Result<Vec<Option<PreparedPartitionInterventionSource>>, eredu_nn::Error>,
        >(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
