//! Original companion frames use the existing receipt, vote and Host worker.
use super::*;

/// One actual original before/after source and the same partition capture
/// program used by ordinary selections. Its optional causal declaration is
/// retained from actual loaded paths; it grants no numerical source or callback.
#[derive(Debug)]
pub struct PreparedPartitionInterventionEvidence<'t,T:PartitionCaptureTransport> {
    pub(super) program:PreparedPartitionCaptureProgram<'t,T>,
    selection:Option<crate::layered::PreparedCaptureSelection>,
    original:OriginalInterventionSource,
    operation:usize,
    prefill:bool,
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionInterventionEvidence<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    pub fn new(original:&OriginalInterventionSource,operation:usize,
        program:PreparedPartitionCaptureProgram<'t,T>,selection:Option<crate::layered::PreparedCaptureSelection>)
        ->Result<Self,PartitionCaptureProgramError> {
        program.metadata.reserve_metadata(size_of::<Self>()*2+size_of::<Option<Self>>()
            +size_of::<Result<Self,PartitionCaptureProgramError>>()
            +size_of::<(&OriginalInterventionSource,usize)>()
            +size_of::<Option<crate::layered::BoundCaptureSelection<'_>>>())
            .map_err(|cause|program.error(cause.into()))?;
        let companion=original.plan().evidence(operation).ok_or_else(||program.error(Cause::Source("original evidence companion is absent")))?;
        let prefill=program.context.phase==CapturePhase::Prefill
            && original.plan().admission().points().get(operation)
                .is_some_and(crate::intervention::InterventionPrefillWindow::row_axis);
        if !program.source.same_storage(companion.shared_geometry_source())||program.rows.len()!=2
            ||program.first_epoch.is_some()||program.interventions.is_some()
            ||selection.as_ref().is_some_and(|selected|!selected.source().same_storage(&program.source))
            ||prefill&&selection.is_none() {
            return Err(program.error(Cause::Source("original companion receipt or causal declaration differs")));
        }
        Ok(Self{program,selection,original:original.clone(),operation,prefill})
    }
    pub(super) fn bind_budget(&mut self,parent:&SharedCapturePlan)->Result<(),Cause> {
        use super::super::super::receipt::EvidenceBudgetSource;
        if self.program.projected.len()!=2 || self.program.rows.iter().any(Option::is_some) {
            return Err(Cause::Source("evidence budget requires both original fragment Host sources"));
        }
        for row in &mut self.program.projected {
            let Some(projected::Row::Pending {source:projected::Source::Contiguous(source),..})=row else {
                return Err(Cause::Source("evidence budget has no unspent dense fragment source"));
            };
            if source.evidence_budget.is_some() {return Err(Cause::Source("evidence budget source already bound"));}
            self.program.metadata.reserve_metadata(EvidenceBudgetSource::controls()
                .ok_or(Cause::Source("evidence budget controls overflow"))?)?;
            source.evidence_budget=Some(EvidenceBudgetSource::new(parent,&self.original,self.operation,&self.program.source)
                .map_err(Cause::Source)?);
        }
        Ok(())
    }
    pub(super) fn matches(&self,original:&OriginalInterventionSource,operation:usize,parent:&PreparedPartitionCaptureProgram<'t,T>)->bool {
        let a=&self.program.context;let b=&parent.context;
        self.original.same_source(original)&&self.operation==operation
            &&std::ptr::eq(self.program.transport,parent.transport)
            &&a.artifact_identity==b.artifact_identity&&a.execution_identity==b.execution_identity
            &&a.overlay_identity==b.overlay_identity&&a.run_identity==b.run_identity
            &&a.phase==b.phase&&a.prediction==b.prediction&&a.invocation==b.invocation
            &&self.program.interventions.is_none()
    }
}
/// The same original companion frame has two distinct points in each chunk:
/// remote source progress before execution, then actual hook completion after it.
#[derive(Clone, Copy)]
enum PrefillEvidenceProgress { Receivers, Complete }
impl<'t,T:PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    fn evidence_controls(&self)->Result<(),PartitionCaptureProgramError> {
        let parts=[
            size_of::<crate::working_memory::PartitionInterventionEvidenceFrame<'_>>()*2,
            size_of::<Result<crate::working_memory::PartitionInterventionEvidenceFrame<'_>,crate::working_memory::CaptureRunHostError>>(),
            size_of::<Result<(),crate::working_memory::CaptureRunHostError>>(),
            size_of::<Result<(),PartitionCaptureProgramError>>()*2,
            size_of::<(&mut Self,&mut ScheduledCaptureStep<'_>,DistributedCommitEpoch,&mut CaptureLedger)>(),
            size_of::<(&mut Self,&mut ScheduledCaptureStep<'_>,crate::layered::BoundCaptureSelection<'_>,&crate::prefill::PrefillChunk,PrefillEvidenceProgress)>(),
            size_of::<(SharedCapturePlan,WorkspaceMetadataFunding)>(),
            size_of::<(&SharedCapturePlan,&WorkspaceMetadataFunding)>(),
            size_of::<Option<&mut dyn ScheduledPartitionCapture>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<crate::layered::PreparedCaptureSelectionError>()
                .ok_or_else(||self.error(Cause::Source("evidence declaration controls overflow")))?,
        ];
        let bytes=parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
            .ok_or_else(||self.error(Cause::Source("evidence frame controls overflow")))?;
        self.metadata.reserve_metadata(bytes).map_err(|cause|self.error(cause.into()))
    }
    pub(super) fn prepare_intervention_evidence(&mut self,epoch:DistributedCommitEpoch,
        frame:&mut ScheduledCaptureStep<'_>,ledger:&mut CaptureLedger)->Result<(),PartitionCaptureProgramError> {
        if self.interventions.as_ref().is_none_or(|rows|rows.evidence.iter().all(Option::is_none)){return Ok(());}
        self.evidence_controls()?;
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let prefill=frame.prefill_geometry();
        let result=(||->Result<(),PartitionCaptureProgramError>{
            let Some(rows)=&mut self.interventions else{return Ok(())};
            for (index,child) in rows.evidence.iter_mut().enumerate() {
                let Some(child)=child else{continue};
                let mut loan=frame.take_partition_intervention_evidence(index).map_err(|cause|error(cause.into()))?;
                let source=child.program.source.clone();let phase=child.program.context.phase;let prediction=child.program.context.prediction;
                let result:Result<(),PartitionCaptureProgramError>=(|| {
                    if child.prefill {
                        let inference=prefill.ok_or_else(||error(Cause::Source("evidence prefill lacks its actual parent schedule")))?;
                        child.selection.as_ref().ok_or_else(||error(Cause::Source("evidence has no causal source")))?
                            .bind_geometry(inference).map_err(|cause|PartitionCaptureProgramError::local(cause,source.clone(),metadata.clone()))?;
                        loan.frame_mut().prepare_existing_prefill_with_progression(inference).map_err(|cause|error(cause.into()))?;
                    }
                    child.program.prepare(&source,phase,prediction,epoch,loan.frame_mut(),ledger)
                })();
                let restore=frame.return_partition_intervention_evidence(loan);
                result?;
                restore.map_err(|cause|error(cause.into()))?;
            }
            Ok(())
        })();result
    }
    pub(in crate::capture::partition::funded) fn coordinate_intervention_evidence(&mut self,epoch:DistributedCommitEpoch,ledger:&CaptureLedger)
        ->Result<(),PartitionCaptureProgramError> {
        if let Some(rows)=&mut self.interventions {
            for child in rows.evidence.iter_mut().flatten() {child.program.coordinate(epoch,ledger)?;}
        }
        Ok(())
    }
    pub(in crate::capture::partition::funded) fn intervention_evidence(&mut self,index:usize)
        ->Result<Option<&mut (dyn ScheduledPartitionCapture + '_)>,PartitionCaptureProgramError> {
        if !self.coordination_complete||self.delivered {return Err(self.error(Cause::Source("evidence callback is outside its coordinated frame")));}
        let Some(rows)=&mut self.interventions else{return Ok(None)};
        Ok(rows.evidence.get_mut(index).and_then(Option::as_mut)
            .map(|child|&mut child.program as &mut dyn ScheduledPartitionCapture))
    }
    pub(in crate::capture::partition::funded) fn progress_intervention_evidence(&mut self,frame:&mut ScheduledCaptureStep<'_>,
        bound:crate::layered::BoundCaptureSelection<'_>,chunk:&crate::prefill::PrefillChunk)->Result<(),PartitionCaptureProgramError> {
        self.advance_intervention_evidence(frame,bound,chunk,PrefillEvidenceProgress::Receivers)
    }
    pub(in crate::capture::partition::funded) fn complete_intervention_evidence(&mut self,frame:&mut ScheduledCaptureStep<'_>,
        bound:crate::layered::BoundCaptureSelection<'_>,chunk:&crate::prefill::PrefillChunk)->Result<(),PartitionCaptureProgramError> {
        self.advance_intervention_evidence(frame,bound,chunk,PrefillEvidenceProgress::Complete)
    }
    fn advance_intervention_evidence(&mut self,frame:&mut ScheduledCaptureStep<'_>,
        bound:crate::layered::BoundCaptureSelection<'_>,chunk:&crate::prefill::PrefillChunk,progress:PrefillEvidenceProgress)
        ->Result<(),PartitionCaptureProgramError> {
        if self.interventions.as_ref().is_none_or(|rows|rows.evidence.iter().all(Option::is_none)){return Ok(());}
        self.evidence_controls()?;
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let result=(||->Result<(),PartitionCaptureProgramError>{
            let Some(rows)=&mut self.interventions else{return Ok(())};
            for (index,child) in rows.evidence.iter_mut().enumerate() {
                let Some(child)=child else{continue};
                if !child.prefill {continue;}
                let selected=child.selection.as_ref().ok_or_else(||error(Cause::Source("prefill evidence has no retained causal declaration")))?;
                if !selected.paths().same_storage(bound.selection().paths()) {return Err(error(Cause::Source("prefill evidence paths differ from the loaded invocation")));}
                let bound=selected.bind_geometry(bound.geometry()).map_err(|cause|
                    PartitionCaptureProgramError::local(cause,source.clone(),metadata.clone()))?;
                let mut loan=frame.take_partition_intervention_evidence(index).map_err(|cause|error(cause.into()))?;
                let result:Result<(),PartitionCaptureProgramError>=(|| {
                    match progress {
                        PrefillEvidenceProgress::Receivers=>child.program.remote_prefill(loan.frame_mut(),bound,chunk)?,
                        PrefillEvidenceProgress::Complete=>{
                            let ordinal=chunk.input.start/bound.geometry().prefill_chunk_positions;
                            loan.frame_mut().complete_prefill_chunk(ordinal).map_err(|cause|error(cause.into()))?;
                            if chunk.input.end==bound.geometry().input_positions {
                                loan.frame_mut().finish_local_prefill_targets().map_err(|cause|error(cause.into()))?;
                            }
                        }
                    }
                    Ok(())
                })();
                let restore=frame.return_partition_intervention_evidence(loan);
                result?;
                restore.map_err(|cause|error(cause.into()))?;
            }
            Ok(())
        })();result
    }
    pub(in crate::capture::partition::funded) fn deliver_intervention_evidence(&mut self,frame:&mut ScheduledCaptureStep<'_>)
        ->Result<(),PartitionCaptureProgramError> {
        if self.interventions.as_ref().is_none_or(|rows|rows.evidence.iter().all(Option::is_none)){return Ok(());}
        self.evidence_controls()?;
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let result=(||->Result<(),PartitionCaptureProgramError>{
            let Some(rows)=&mut self.interventions else{return Ok(())};
            for (index,child) in rows.evidence.iter_mut().enumerate() {
                let Some(child)=child else{continue};
                let mut loan=frame.take_partition_intervention_evidence(index).map_err(|cause|error(cause.into()))?;
                let result:Result<(),PartitionCaptureProgramError>=(|| {
                    child.program.deliver(loan.frame_mut())?;
                    if loan.frame().prefill_geometry().is_some() {
                        loan.frame_mut().finish_prefill_targets().map_err(|cause|error(cause.into()))?;
                    }
                    Ok(())
                })();
                let restore=frame.return_partition_intervention_evidence(loan);
                result?;
                restore.map_err(|cause|error(cause.into()))?;
            }
            Ok(())
        })();result
    }
}
