//! Contiguous rows inside the existing scheduled receipt program.
use super::*;
use crate::working_memory::PreparedPartitionFragmentHostFunding;

pub(super) enum Row<'t,T:PartitionCaptureTransport> {
    Pending{source:Source,host:PreparedPartitionFragmentHostFunding},
    Ready(PreparedPartitionContiguousRow<'t,T>),
    Skipped{producer:usize,usage:CaptureUsage,reason:CaptureSkipReason,cause:PartitionCaptureProgramError},
    Spent,
}
pub(super) enum Source { Contiguous(PreparedPartitionContiguousSource), Routed(PreparedPartitionRoutedSource) }
impl Source {
    fn matches_context(&self,context:&PartitionCaptureContext)->bool {match self {
        Self::Contiguous(source)=>source.matches_context(context),Self::Routed(source)=>source.matches_context(context),
    }}
    fn matches_program(&self,source:&SharedCapturePlan,index:usize)->bool {match self {
        Self::Contiguous(value)=>value.matches_program(source,index),Self::Routed(value)=>value.matches_program(source,index),
    }}
    fn prepare_selected_with_host<'t,T:PartitionCaptureTransport>(self,transport:&'t T,context:&PartitionCaptureContext,
        epoch:DistributedCommitEpoch,host:PreparedPartitionFragmentHostFunding,limits:PartitionCaptureReceiptLimits,ledger:&mut CaptureLedger)
        ->Result<PreparedPartitionContiguousRow<'t,T>,contiguous::PreparationFailure>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {match self {
        Self::Contiguous(source)=>source.prepare_selected_with_host(transport,context,epoch,host,limits,ledger),
        Self::Routed(source)=>source.prepare_selected_with_host(transport,context,epoch,host,limits,ledger),
    }}
}
#[derive(Debug,thiserror::Error)]
#[error("projected Host source exceeds the admitted receipt program limits")]
struct HeldHostLimits { _host:PreparedPartitionFragmentHostFunding }
#[derive(Debug,thiserror::Error)]
#[error("projected skip completion: {cause}")]
struct HeldSkip<E:std::error::Error+'static> {
    #[source] cause:E,
    _preparation:PartitionCaptureProgramError,
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    /// Attach the actual source and already protected original Host destination
    /// for one actual projected selection. Complete rows use the same program;
    /// this owner does not supply native permission or a transport replacement.
    pub fn with_contiguous_row(self,index:usize,source:PreparedPartitionContiguousSource,
        host:PreparedPartitionFragmentHostFunding)->Result<Self,PartitionCaptureProgramError> {
        self.with_projected_row(index,Source::Contiguous(source),host)
    }
    /// Attach the sparse source to the same original receipt/vote/delivery row.
    pub fn with_routed_row(mut self,index:usize,source:PreparedPartitionRoutedSource,
        host:PreparedPartitionFragmentHostFunding)->Result<Self,PartitionCaptureProgramError> {
        self.prepare_routed_hook_storage()?;
        self.with_projected_row(index,Source::Routed(source),host)
    }
    fn with_projected_row(mut self,index:usize,source:Source,
        host:PreparedPartitionFragmentHostFunding)->Result<Self,PartitionCaptureProgramError> {
        self.metadata.reserve_metadata(control_bytes::<T>().ok_or_else(||self.error(Cause::Source("projected controls overflow")))?)
            .map_err(|e|self.error(e.into()))?;
        if self.first_epoch.is_some()||!source.matches_context(&self.context)
            ||index>=self.rows.len()||self.rows[index].is_some()||!source.matches_program(&self.source,index)
            ||self.projected.get(index).is_some_and(Option::is_some) {
            return Err(self.error(Cause::Source("projected row differs from the unprepared program")));
        }
        if self.projected.is_empty(){
            self.projected=self.metadata.metadata_vec(self.rows.len()).map_err(|e|self.error(e.into()))?;
            self.projected.resize_with(self.rows.len(),||None);
        }
        self.projected[index]=Some(Row::Pending{source,host});Ok(self)
    }
    fn charge_projected(&self)->Result<(),PartitionCaptureProgramError> {
        self.metadata.reserve_metadata(control_bytes::<T>().ok_or_else(||self.error(Cause::Source("projected controls overflow")))?)
            .map_err(|e|self.error(e.into()))
    }
    pub(super) fn prepare_projected(&mut self,index:usize,epoch:DistributedCommitEpoch,
        frame:&mut ScheduledCaptureStep<'_>,ledger:&mut CaptureLedger)
        ->Result<bool,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(false);}
        self.charge_projected()?;
        let Some(slot)=self.projected.get_mut(index) else{return Ok(false)};
        let Some(row)=slot.take() else{return Ok(false)};
        *slot=Some(Row::Spent);
        let Row::Pending{source,host}=row else{return Err(self.error(Cause::Source("projected preparation is spent")))};
        // Preserve the exact per-selection limits that constructed the original
        // Host owner. The program can cover other selections with wider limits,
        // but it cannot enlarge or replace this retained receipt source.
        let limits=host.original_receipt_limits();
        if limits.max_producers>self.limits.max_producers
            ||limits.max_fragments>self.limits.max_fragments
            ||limits.max_record_bytes>self.limits.max_record_bytes {
            return Err(PartitionCaptureProgramError::local(HeldHostLimits{_host:host},
                self.source.clone(),self.metadata.clone()));
        }
        match source.prepare_selected_with_host(self.transport,&self.context,epoch,host,limits,ledger) {
            Ok(row)=>self.projected[index]=Some(Row::Ready(row)),
            Err(failure)=>{
                let Some(reason)=self.allowed_skip(failure.reason) else{return Err(failure.cause)};
                let Some((producer,usage))=failure.descriptor else{return Err(failure.cause)};
                if let Err(cause)=frame.record_partition_skip(index,reason.clone()) {
                    return Err(PartitionCaptureProgramError::local(HeldSkip{cause,_preparation:failure.cause},
                        self.source.clone(),self.metadata.clone()));
                }
                self.skipped[index]=Some(reason.clone());
                self.projected[index]=Some(Row::Skipped{producer,usage,reason,cause:failure.cause});
            }
        }
        Ok(true)
    }
    pub(super) fn coordinate_projected(&mut self,index:usize,coordination:&mut PreparedPartitionCaptureCoordination<'t,T>,ledger:&CaptureLedger)
        ->Result<bool,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(false);}
        self.charge_projected()?;
        if matches!(self.projected[index],Some(Row::Skipped{..})) {
            let Some(Row::Skipped{producer,usage,reason,cause:preparation})=self.projected[index].take() else{unreachable!("checked skipped row")};
            self.projected[index]=Some(Row::Spent);
            // No receipt or scalar is forged for a skipped row. The existing
            // Source and Coordination workers compare the actual retained
            // rank, estimate, reason and spent ledger on every participant.
            let result=coordination.coordinate_source(index,None,producer,None,Some(&reason),usage,ledger)
                .and_then(|_|coordination.include_skip(index,&reason));
            if let Err(cause)=result {
                return Err(PartitionCaptureProgramError::local(HeldSkip{cause,_preparation:preparation},
                    self.source.clone(),self.metadata.clone()));
            }
            self.projected[index]=Some(Row::Skipped{producer,usage,reason,cause:preparation});
            return Ok(true);
        }
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(false)};
        let Row::Ready(row)=slot else{return Err(self.error(Cause::Source("projected source is not ready")))};
        if row.is_routed() {
            let vote=match coordination.coordinate_routed_source(index,row.receipt()?,row.source_present(),row.local_dtype(),row.source_usage(),ledger){
                Ok(vote)=>vote,Err(cause)=>return Err(row.retain_vote_failure(cause)),
            };
            let dtype=vote.dtype.clone();row.bind_routed_vote(vote.dtype,vote.ranks)?;
            coordination.include(row.receipt()?,dtype,row.source_usage()).map_err(|cause|row.retain_vote_failure(cause))?;
            return Ok(true);
        }
        let dtype=match coordination.coordinate_contiguous_source(index,row.receipt()?,row.local_dtype(),row.source_usage(),ledger){
            Ok(dtype)=>dtype,Err(cause)=>return Err(row.retain_vote_failure(cause)),
        };
        row.bind_voted_scalar(dtype.clone())?;
        coordination.include(row.receipt()?,dtype,row.source_usage()).map_err(|cause|row.retain_vote_failure(cause))?;
        Ok(true)
    }
    pub(super) fn projected_epoch(&self)->Result<DistributedCommitEpoch,PartitionCaptureProgramError>{
        self.charge_projected()?;
        if !self.coordination_complete||self.delivered{return Err(self.error(Cause::Source("projected hook is outside its coordinated frame")));}
        self.last_epoch.ok_or_else(||self.error(Cause::Source("projected hook has no actual epoch")))
    }
    pub(super) fn take_projected_hook(&mut self,index:usize,frame:&mut ScheduledCaptureStep<'_>,first:bool)
        ->Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(None);}
        let epoch=self.projected_epoch()?;
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(None)};
        if matches!(slot,Row::Skipped{..}){return Ok(None);}
        let Row::Ready(row)=slot else{return Err(self.error(Cause::Source("projected hook owner is unavailable")))};
        if row.geometry().is_none(){return Err(self.error(Cause::Source("decode row cannot issue a prefill hook")));}
        if first {row.reserve_target(frame)?;}
        row.take_local(row.next_chunk(),epoch)
    }
    pub(super) fn take_projected_receiver(&mut self,index:usize,inference:eredu_core::InferenceGeometry,chunk:u64)
        ->Result<Option<PartitionPrefillReceiverSource>,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(None);}
        let epoch=self.projected_epoch()?;
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(None)};
        if matches!(slot,Row::Skipped{..}){return Ok(None);}
        let Row::Ready(row)=slot else{return Err(self.error(Cause::Source("projected receiver owner is unavailable")))};
        if row.geometry()!=Some(inference){return Err(self.error(Cause::Source("projected receiver geometry differs")));}
        row.take_receiver(chunk,epoch)
    }
    pub(super) fn take_projected_invocation(&mut self,index:usize)
        ->Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(None);}
        let epoch=self.projected_epoch()?;
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(None)};
        if matches!(slot,Row::Skipped{..}){return Ok(None);}
        let Row::Ready(row)=slot else{return Err(self.error(Cause::Source("decode source is not ready")))};
        if row.geometry().is_some(){return Err(self.error(Cause::Source("prefill row cannot issue a decode hook")));}
        row.take_local(0,epoch)
    }
    pub(super) fn take_projected_invocation_receiver(&mut self,index:usize)
        ->Result<Option<super::PartitionInvocationReceiverSource>,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(None);}
        let epoch=self.projected_epoch()?;
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(None)};
        if matches!(slot,Row::Skipped{..}){return Ok(None);}
        let Row::Ready(row)=slot else{return Err(self.error(Cause::Source("decode source is not ready")))};
        row.take_invocation_receiver(epoch)
    }
    pub(super) fn progress_projected_receiver(&mut self,index:usize,frame:&mut ScheduledCaptureStep<'_>,
        bound:crate::layered::BoundCaptureSelection<'_>,chunk:&crate::prefill::PrefillChunk)
        ->Result<bool,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(false);}
        self.charge_projected()?;
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Ok(false)};
        if matches!(slot,Row::Skipped{..}){return Ok(true);}
        let Row::Ready(projected)=slot else{return Err(error(Cause::Source("projected progress owner is unavailable")))};
        if projected.produces() || projected.is_routed()&&projected.source_present(){return Ok(true);}
        if projected.geometry()!=Some(bound.geometry()){return Err(error(Cause::Source("projected progress geometry differs")));}
        let path=&source.admission().plan().selections[index].path;
        let decision=if projected.is_routed(){frame.begin_routed_prefill_batch(index,chunk,path)}else{frame.begin_prefill_hook(index,chunk,path)}
            .map_err(|e|error(e.into()))?;
        if decision==crate::capture::CapturePrefillHookDecision::Ignore{return Ok(true);}
        if decision==crate::capture::CapturePrefillHookDecision::First{projected.reserve_target(frame)?;}
        let policy=crate::capture::CapturePrefillObservationPolicy::from_bound(bound).map_err(|e|error(e.into()))?;
        let row=policy.row(index).map_err(|e|error(e.into()))?;
        let k=chunk.input.start/bound.geometry().prefill_chunk_positions;
        if projected.is_routed(){
            let plan=row.routed_plan().ok_or_else(||error(Cause::Source("sparse progression has no original plan")))?;
            let fragment=plan.fragment(k).map_err(|e|error(Cause::Progress(e.into())))?;
            frame.finish_partition_routed_prefill_hook(index,&fragment).map_err(|e|error(e.into()))?;
            return Ok(true);
        }
        if let Some(plan)=row.transform_plan(){
            let fragment=plan.fragment(k).map_err(|e|error(Cause::Progress(e.into())))?;
            match plan.selection().transform {
                CaptureTransform::Summary=>frame.finish_summary_prefill_hook(index,&fragment),
                CaptureTransform::Histogram{..}=>frame.finish_histogram_prefill_hook(index,&fragment),
                _=>return Err(error(Cause::Source("projected reduction differs"))),
            }.map_err(|e|error(e.into()))?;
        }else{
            let assembly=row.assembly().ok_or_else(||error(Cause::Source("projected global assembly is absent")))?;
            let fragment=assembly.fragment(k).map_err(|e|error(Cause::Progress(e.into())))?;
            frame.finish_prefill_hook(index,&fragment).map_err(|e|error(e.into()))?;
        }
        Ok(true)
    }
    pub(super) fn deliver_projected(&mut self,index:usize,frame:&mut ScheduledCaptureStep<'_>)
        ->Result<bool,PartitionCaptureProgramError> {
        if !self.projected.get(index).is_some_and(Option::is_some){return Ok(false);}
        self.charge_projected()?;
        let Some(slot)=self.projected.get_mut(index) else{return Ok(false)};
        let Some(row)=slot.take() else{return Ok(false)};*slot=Some(Row::Spent);
        if matches!(&row,Row::Skipped{..}){return Ok(true);}
        let Row::Ready(row)=row else{return Err(self.error(Cause::Source("projected delivery is spent")))};
        let prefill=row.geometry().is_some();let delivery=row.finish()?;
        let result=if prefill{delivery.deliver_into_prefill_frame(frame)}else{delivery.deliver_into_invocation_frame(frame)};
        result.map_err(|e|PartitionCaptureProgramError::local(e,self.source.clone(),self.metadata.clone()))?;
        Ok(true)
    }
}
fn control_bytes<T:PartitionCaptureTransport>()->Option<usize>{
    let parts=[size_of::<Source>(),size_of::<contiguous::PreparationFailure>()*2,
        size_of::<PartitionCaptureReceiptLimits>()*3,size_of::<HeldHostLimits>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldHostLimits>()?,
        size_of::<Result<PreparedPartitionContiguousRow<'_,T>,contiguous::PreparationFailure>>(),
        size_of::<(usize,CaptureUsage)>(),size_of::<Option<(usize,CaptureUsage)>>(),
        size_of::<CaptureSkipReason>()*2,size_of::<Option<CaptureSkipReason>>(),
        size_of::<HeldSkip<crate::working_memory::CaptureRunHostError>>(),
        size_of::<HeldSkip<PartitionCaptureCoordinationError>>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldSkip<crate::working_memory::CaptureRunHostError>>()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<HeldSkip<PartitionCaptureCoordinationError>>()?,
        size_of::<Row<'_,T>>()*2,size_of::<Option<Row<'_,T>>>(),
        size_of::<Vec<Option<Row<'_,T>>>>(),size_of::<PreparedPartitionCaptureProgram<'_,T>>()*2,
        size_of::<PreparedPartitionContiguousSource>(),size_of::<PreparedPartitionFragmentHostFunding>(),
        size_of::<PreparedPartitionContiguousRow<'_,T>>(),size_of::<Result<PreparedPartitionCaptureProgram<'_,T>,PartitionCaptureProgramError>>(),
        size_of::<Result<bool,PartitionCaptureProgramError>>(),size_of::<Result<Option<PartitionCaptureLocalHook>,PartitionCaptureProgramError>>(),
        size_of::<Result<Option<PartitionPrefillReceiverSource>,PartitionCaptureProgramError>>(),
        size_of::<Result<Option<super::PartitionInvocationReceiverSource>,PartitionCaptureProgramError>>(),
        super::PartitionInvocationReceiverSource::control_bytes()?,
        size_of::<(&mut PreparedPartitionCaptureProgram<'_,T>,usize,&mut ScheduledCaptureStep<'_>,bool)>(),
        size_of::<(usize,DistributedCommitEpoch,eredu_core::InferenceGeometry,bool)>(),
        size_of::<crate::capture::CapturePrefillObservationPolicy<'_>>(),size_of::<crate::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<eredu_core::capture::CapturePrefillTransformFragment<'_,'_>>(),size_of::<Option<&eredu_core::capture::CapturePrefillTransformPlan<'_>>>(),
        size_of::<Option<TensorDtype>>(),size_of::<CaptureUsage>(),size_of::<PartitionCaptureProgramError>(),
        size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>(),size_of::<(&SharedCapturePlan,&HostMetadataFunding)>(),
        size_of::<eredu_core::capture::CapturePrefillFragment<'_,'_>>(),
        size_of::<crate::capture::CapturePrefillProgressError>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
