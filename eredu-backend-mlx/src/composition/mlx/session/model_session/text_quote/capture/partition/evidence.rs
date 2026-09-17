//! Attach original two-side Host owners to the existing operation transaction.
use super::*;
use eredu_core::{capture::InterventionEvidenceSide,intervention::InterventionEvidence};
use eredu_runtime::capture::partition::PreparedPartitionInterventionEvidence;
use crate::composition::mlx::session::intervention::PreparedTextInterventions;
use eredu_nn::workspace::WorkspaceMetadataFunding;

pub(super) fn control_bytes()->Option<usize> {
    let parts=[size_of::<host::evidence::Operation>()*2,
        size_of::<Vec<Option<host::evidence::Operation>>>(),
        size_of::<std::cell::RefMut<'_,Option<host::evidence::Hosts>>>(),
        size_of::<PreparedPartitionInterventionEvidence<'_,OriginalCaptureTransport>>()*2,
        size_of::<Vec<Option<PreparedPartitionInterventionEvidence<'_,OriginalCaptureTransport>>>>(),
        size_of::<Vec<Option<host::HostRow>>>(),size_of::<Option<host::HostRow>>(),
        size_of::<[PreparedPartitionCaptureRow<'_>;2]>(),size_of::<[InterventionEvidenceSide;2]>(),
        size_of::<std::array::IntoIter<InterventionEvidenceSide,2>>(),
        size_of::<(usize,Option<WorkspaceFloatingType>)>(),
        size_of::<(&OriginalCaptureTransport,&str,&str,eredu_runtime::CommunicationSessionIdentity,Option<&str>,
            &SharedCapturePlan,&PreparedTextInterventions,CapturePhase,u64,&mut [Option<host::evidence::Operation>],&WorkspaceMetadataFunding)>(),
        size_of::<Result<Vec<Option<PreparedPartitionInterventionEvidence<'_,OriginalCaptureTransport>>>,Error>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'t>(transport:&'t OriginalCaptureTransport,artifact:&str,execution:&str,
    setup:eredu_runtime::CommunicationSessionIdentity,overlay:Option<&str>,parent:&SharedCapturePlan,
    interventions:&PreparedTextInterventions,phase:CapturePhase,prediction:u64,
    hosts:&mut [Option<host::evidence::Operation>],metadata:&WorkspaceMetadataFunding)
    ->Result<Vec<Option<PreparedPartitionInterventionEvidence<'t,OriginalCaptureTransport>>>,Error> {
    metadata.reserve_metadata(control_bytes().ok_or_else(||memory(WorkingMemoryError::Overflow))?)?;
    let original=interventions.source();let plan=original.plan().admission();
    if !hosts.is_empty()&&hosts.len()!=plan.plan().operations.len(){return Err(memory(WorkingMemoryError::IdentityMismatch));}
    let mut output=metadata.metadata_vec(plan.plan().operations.len())?;
    for (operation,entry) in plan.plan().operations.iter().enumerate() {
        if !entry.schedule.includes(phase,prediction)||entry.evidence==InterventionEvidence::None {output.push(None);continue;}
        let owner=hosts.get_mut(operation).and_then(Option::take).ok_or_else(unknown)?;
        let companion=original.plan().evidence(operation).ok_or_else(unknown)?.shared_geometry_source();
        if !owner.source.same_storage(companion)||owner.hosts.len()!=2||owner.hosts.iter().any(Option::is_none){
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let limits=PartitionCaptureReceiptLimits{max_producers:setup.participant_count(),
            max_fragments:owner.hosts.iter().flatten().map(|row|row.fragment_limit()).max().ok_or_else(unknown)?,
            max_record_bytes:parent.admission().plan().limits.per_step.encoded_bytes};
        // Same setup labels. The atomic parent attachment below binds the actual
        // shared first-epoch/run owner and the private original budget source.
        let context=PreparedPartitionCaptureRunIdentity::prepare_host_context(companion,artifact,execution,setup,overlay,phase,prediction,metadata)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        let selected=[PreparedPartitionCaptureRow::Contiguous;2];
        let mut program=PreparedPartitionCaptureProgram::new_selected(transport,companion,&context,&selected,limits,metadata)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        for (index,host) in owner.hosts.into_iter().enumerate() {
            let side=match index {0=>InterventionEvidenceSide::Before,1=>InterventionEvidenceSide::After,_=>unreachable!("two original sides")};
            let scalar=interventions.partition_evidence_scalar(phase,prediction,operation,side,metadata)?;
            let (source,host)=host.ok_or_else(unknown)?.bind(companion,transport.capture_rank(),scalar,metadata)?;
            let host::BoundSource::Contiguous(source)=source else{return Err(memory(WorkingMemoryError::IdentityMismatch))};
            program=program.with_contiguous_row(index,source,host)
                .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
        }
        output.push(Some(PreparedPartitionInterventionEvidence::new(original,operation,program,owner.selection)
            .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?));
    }
    Ok(output)
}
