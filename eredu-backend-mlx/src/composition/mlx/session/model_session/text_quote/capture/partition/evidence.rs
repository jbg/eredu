//! Attach original two-side Host owners to the existing operation transaction.
use super::*;
use crate::composition::mlx::session::intervention::{
    PreparedModelInterventions, PreparedTextInterventions,
};
use eredu_core::{capture::InterventionEvidenceSide, intervention::InterventionEvidence};
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::capture::partition::PreparedPartitionInterventionEvidence;
use eredu_runtime::working_memory::OriginalInterventionSource;

pub(super) fn control_bytes<T: NativePartitionCaptureTransport>() -> Option<usize> {
    source_controls::<T, PreparedTextInterventions>()
}
pub(super) fn model_control_bytes<T: NativePartitionCaptureTransport>() -> Option<usize> {
    source_controls::<T, PreparedModelInterventions>()
}
fn source_controls<T: NativePartitionCaptureTransport, I: EvidenceSource>() -> Option<usize> {
    let parts = [
        size_of::<host::evidence::Operation>() * 2,
        size_of::<Option<PartitionCaptureInvocation>>(),
        size_of::<Option<eredu_core::capture::CaptureInvocationShape>>(),
        size_of::<Option<eredu_core::capture::PartitionCaptureInvocationWindow>>(),
        size_of::<Vec<Option<host::evidence::Operation>>>(),
        size_of::<std::cell::RefMut<'_, Option<host::evidence::Hosts>>>(),
        size_of::<PreparedPartitionInterventionEvidence<'_, T>>() * 2,
        size_of::<Vec<Option<PreparedPartitionInterventionEvidence<'_, T>>>>(),
        size_of::<Vec<Option<host::HostRow>>>(),
        size_of::<Option<host::HostRow>>(),
        size_of::<[PreparedPartitionCaptureRow<'_>; 2]>(),
        size_of::<[InterventionEvidenceSide; 2]>(),
        size_of::<std::array::IntoIter<InterventionEvidenceSide, 2>>(),
        size_of::<(usize, Option<WorkspaceFloatingType>)>(),
        size_of::<(
            &T,
            &str,
            &str,
            eredu_runtime::CommunicationSessionIdentity,
            Option<&str>,
            &SharedCapturePlan,
            &I,
            CapturePhase,
            u64,
            &mut [Option<host::evidence::Operation>],
            &HostMetadataFunding,
        )>(),
        size_of::<Result<Vec<Option<PreparedPartitionInterventionEvidence<'_, T>>>, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'t, T: NativePartitionCaptureTransport, I: EvidenceSource>(
    transport: &'t T,
    artifact: &str,
    execution: &str,
    setup: eredu_runtime::CommunicationSessionIdentity,
    overlay: Option<&str>,
    parent: &SharedCapturePlan,
    interventions: &I,
    phase: CapturePhase,
    prediction: u64,
    hosts: &mut [Option<host::evidence::Operation>],
    metadata: &HostMetadataFunding,
) -> Result<Vec<Option<PreparedPartitionInterventionEvidence<'t, T>>>, Error>
where
    <T::Completion as eredu_core::Completion>::Error: Send + Sync + 'static,
{
    metadata.reserve_metadata(
        source_controls::<T, I>().ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
    )?;
    let original = interventions.original();
    let invocation = interventions.invocation();
    if invocation.is_some_and(|value| value.phase != phase || value.prediction != prediction) {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    let logical = invocation
        .map(|value| value.logical())
        .transpose()
        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
    let plan = original.plan().admission();
    if !hosts.is_empty() && hosts.len() != plan.plan().operations.len() {
        return Err(memory(WorkingMemoryError::IdentityMismatch));
    }
    let mut output = metadata.metadata_vec(plan.plan().operations.len())?;
    for (operation, entry) in plan.plan().operations.iter().enumerate() {
        if !entry.schedule.includes(phase, prediction)
            || entry.evidence == InterventionEvidence::None
            || !interventions.reached(operation, metadata)?
        {
            output.push(None);
            continue;
        }
        let owner = hosts
            .get_mut(operation)
            .and_then(Option::take)
            .ok_or_else(unknown)?;
        let companion = original
            .plan()
            .evidence(operation)
            .ok_or_else(unknown)?
            .shared_geometry_source();
        if !owner.source.same_storage(companion)
            || owner.hosts.len() != 2
            || owner.hosts.iter().any(Option::is_none)
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let limits = PartitionCaptureReceiptLimits {
            max_producers: setup.participant_count(),
            max_fragments: owner
                .hosts
                .iter()
                .flatten()
                .map(|row| row.fragment_limit())
                .max()
                .ok_or_else(unknown)?,
            max_record_bytes: parent.admission().plan().limits.per_step.encoded_bytes,
        };
        // Same setup labels. The atomic parent attachment below binds the actual
        // shared first-epoch/run owner and the private original budget source.
        let context = PreparedPartitionCaptureRunIdentity::prepare_host_context_at(
            companion,
            artifact,
            execution,
            setup,
            overlay,
            phase,
            prediction,
            logical,
            invocation.and_then(PartitionCaptureInvocation::receipt_window),
            metadata,
        )
        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        let selected = [PreparedPartitionCaptureRow::Contiguous; 2];
        let mut program = PreparedPartitionCaptureProgram::new_selected(
            transport, companion, &context, &selected, limits, metadata,
        )
        .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        for (index, host) in owner.hosts.into_iter().enumerate() {
            let side = match index {
                0 => InterventionEvidenceSide::Before,
                1 => InterventionEvidenceSide::After,
                _ => unreachable!("two original sides"),
            };
            let scalar = interventions.scalar(phase, prediction, operation, side, metadata)?;
            let (source, host) = host.ok_or_else(unknown)?.bind(
                companion,
                transport.capture_rank(),
                scalar,
                metadata,
            )?;
            let host::BoundSource::Contiguous(source) = source else {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            };
            program = program
                .with_contiguous_row(index, source, host)
                .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?;
        }
        output.push(Some(
            PreparedPartitionInterventionEvidence::new(
                original,
                operation,
                program,
                owner.selection,
            )
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?,
        ));
    }
    Ok(output)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;

/// Borrowed preparation facts from the actual admitted Text or Model owner.
/// This adapter creates no intervention or native authority.
pub(super) trait EvidenceSource {
    fn original(&self) -> &OriginalInterventionSource;
    fn invocation(&self) -> Option<PartitionCaptureInvocation>;
    fn reached(&self, operation: usize, metadata: &HostMetadataFunding) -> Result<bool, Error>;
    fn scalar(
        &self,
        phase: CapturePhase,
        prediction: u64,
        operation: usize,
        side: InterventionEvidenceSide,
        metadata: &HostMetadataFunding,
    ) -> Result<Option<WorkspaceFloatingType>, eredu_nn::Error>;
}
impl EvidenceSource for PreparedTextInterventions {
    fn original(&self) -> &OriginalInterventionSource {
        self.source()
    }
    fn invocation(&self) -> Option<PartitionCaptureInvocation> {
        None
    }
    fn reached(&self, _: usize, _: &HostMetadataFunding) -> Result<bool, Error> {
        Ok(true)
    }
    fn scalar(
        &self,
        phase: CapturePhase,
        prediction: u64,
        operation: usize,
        side: InterventionEvidenceSide,
        metadata: &HostMetadataFunding,
    ) -> Result<Option<WorkspaceFloatingType>, eredu_nn::Error> {
        self.partition_evidence_scalar(phase, prediction, operation, side, metadata)
    }
}
impl EvidenceSource for PreparedModelInterventions {
    fn original(&self) -> &OriginalInterventionSource {
        self.source()
    }
    fn invocation(&self) -> Option<PartitionCaptureInvocation> {
        self.invocation_coordinate()
            .map(
                |(phase, prediction, physical, window)| PartitionCaptureInvocation {
                    phase,
                    prediction,
                    physical,
                    window,
                },
            )
    }
    fn reached(&self, operation: usize, metadata: &HostMetadataFunding) -> Result<bool, Error> {
        Ok(self
            .partition_operation_selected(operation)
            .map_err(|cause| Error::Neural(metadata.metadata_source(cause)))?
            && !self
                .partition_evidence_skip_reasons(operation)
                .is_some_and(|sides| sides.iter().all(Option::is_some)))
    }
    fn scalar(
        &self,
        phase: CapturePhase,
        prediction: u64,
        operation: usize,
        side: InterventionEvidenceSide,
        metadata: &HostMetadataFunding,
    ) -> Result<Option<WorkspaceFloatingType>, eredu_nn::Error> {
        self.partition_evidence_scalar(phase, prediction, operation, side, metadata)
    }
}
