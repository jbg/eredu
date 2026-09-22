//! Finite original hook table lent to the existing routed provider invocation.
use super::*;
use crate::capture::{FundedCaptureError,ScheduledCaptureBackend,CapturePrefillHookDecision};
use eredu_core::InferenceGeometry;
#[derive(Debug)]
pub(super) enum Slot { Unused, Active(Option<PartitionCaptureLocalHook>) }
/// Runtime-owned callback table, with no transport methods while its actual
/// receipt and Host owners are detached. It grants no native source authority.
#[derive(Debug)]
pub struct PartitionCaptureRoutedHooks {
    slots:Vec<Slot>,first:usize,prefill:Option<(InferenceGeometry,u64)>,
    source:SharedCapturePlan,metadata:HostMetadataFunding,
}
#[derive(Debug,thiserror::Error)]
#[error("partition routed hook scope: {cause}")]
struct Failure { #[source] cause:PartitionCaptureProgramError,_scope:PartitionCaptureRoutedHooks }
impl PartitionCaptureRoutedHooks {
    pub(crate) fn same_routing(source:&AdmittedCapturePlan,first:usize,index:usize)->bool {
        match (source.points().get(first).map(|p|&p.value_type),source.points().get(index).map(|p|&p.value_type)) {
            (Some(eredu_core::ObservationValueType::RoutedUnits{routing:a,..}),Some(eredu_core::ObservationValueType::RoutedUnits{routing:b,..}))=>a==b,
            _=>false,
        }
    }
    fn failed(self,cause:PartitionCaptureProgramError)->PartitionCaptureProgramError {
        let source=self.source.clone();let metadata=self.metadata.clone();
        PartitionCaptureProgramError::local(Failure{cause,_scope:self},source,metadata)
    }
    pub(crate) fn reject(self,message:&'static str)->PartitionCaptureProgramError {
        let cause=PartitionCaptureProgramError{cause:Cause::Source(message),_source:self.source.clone(),_metadata:self.metadata.clone()};
        self.failed(cause)
    }
    pub(crate) fn begin<T,E:std::error::Error+Send+Sync+'static>(mut self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,invocation:&crate::RoutedUnitInvocation<'_,T>)
        ->Result<Self,PartitionCaptureProgramError> {
        for index in 0..self.slots.len() {
            let Slot::Active(slot)=&mut self.slots[index] else{continue};
            let Some(hook)=slot.take() else{continue};
            match hook.begin_routed(backend,invocation,self.prefill){Ok(hook)=>*slot=Some(hook),Err(cause)=>return Err(self.failed(cause))}
        }
        Ok(self)
    }
    pub(crate) fn observe<T,E:std::error::Error+Send+Sync+'static>(mut self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,batch:&crate::RoutedUnitBatch<'_,T>,effective:bool)
        ->Result<Self,PartitionCaptureProgramError> {
        for index in 0..self.slots.len() {
            if (self.source.admission().points()[index].position==eredu_core::ObservationPosition::AfterIntervention)!=effective {continue;}
            let Slot::Active(slot)=&mut self.slots[index] else{continue};
            let Some(hook)=slot.take() else{continue};
            match hook.observe_routed(backend,batch){Ok(hook)=>*slot=Some(hook),Err(cause)=>return Err(self.failed(cause))}
        }
        Ok(self)
    }
    pub(crate) fn finish<T,E:std::error::Error+Send+Sync+'static>(mut self,success:bool)->Result<Self,PartitionCaptureProgramError> {
        for index in 0..self.slots.len() {
            let Slot::Active(slot)=&mut self.slots[index] else{continue};
            let Some(hook)=slot.take() else{continue};
            match hook.finish_routed::<T,E>(success){Ok(hook)=>*slot=Some(hook),Err(cause)=>return Err(self.failed(cause))}
        }
        Ok(self)
    }
    /// Concrete finite table and transition frames. Backends retain their
    /// independently priced native five-source collection/completion programs.
    pub fn control_bytes<T,E:std::error::Error+Send+Sync+'static>()->Option<usize> {
        let parts=[PartitionCaptureLocalHook::routed_control_bytes::<T,E>()?,Self::table_control_bytes()?,
            size_of::<(&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&crate::RoutedUnitInvocation<'_,T>)>(),
            size_of::<(&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&crate::RoutedUnitBatch<'_,T>,bool)>(),
            size_of::<FundedCaptureError<E>>(),size_of::<Result<Self,PartitionCaptureProgramError>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn table_control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<Slot>()*2,size_of::<Vec<Slot>>(),size_of::<Option<Vec<Slot>>>(),
            size_of::<Failure>(),eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
            size_of::<PartitionCaptureProgramError>(),size_of::<Result<Self,PartitionCaptureProgramError>>(),
            size_of::<Option<(InferenceGeometry,u64)>>(),size_of::<Option<PartitionCaptureLocalHook>>(),
            size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    pub(super) fn prepare_routed_hook_storage(&mut self)->Result<(),PartitionCaptureProgramError> {
        self.metadata.reserve_metadata(PartitionCaptureRoutedHooks::table_control_bytes()
            .ok_or_else(||self.error(Cause::Source("routed hook table controls overflow")))?).map_err(|e|self.error(e.into()))?;
        if self.routed_hooks.is_none(){
            let mut slots=self.metadata.metadata_vec(self.rows.len()).map_err(|e|self.error(e.into()))?;
            slots.resize_with(self.rows.len(),||Slot::Unused);self.routed_hooks=Some(slots);
        }Ok(())
    }
    pub(super) fn has_routed_source(&self,index:usize)->Result<bool,PartitionCaptureProgramError> {
        self.projected_epoch()?;
        let mut any=false;
        for selected in 0..self.rows.len(){
            if !PartitionCaptureRoutedHooks::same_routing(self.source.admission(),index,selected)
                || !self.source.admission().plan().selections[selected].schedule.includes(self.context.phase,self.context.prediction){continue;}
            match self.projected.get(selected).and_then(Option::as_ref){
                Some(projected::Row::Ready(row)) if row.is_routed()=>any=true,
                Some(projected::Row::Skipped{..})=>any=true,
                _=>return Err(self.error(Cause::Source("active sparse point lacks its retained source"))),
            }
        }Ok(any)
    }

    pub(super) fn lend_routed_hooks(&mut self,first:usize,frame:&mut ScheduledCaptureStep<'_>,
        prefill:Option<(&crate::prefill::PrefillChunk,InferenceGeometry)>)->Result<PartitionCaptureRoutedHooks,PartitionCaptureProgramError> {
        if !self.has_routed_source(first)?{return Err(self.error(Cause::Source("provider has no retained sparse source")));}
        let epoch=self.projected_epoch()?;
        self.metadata.reserve_metadata(PartitionCaptureRoutedHooks::table_control_bytes().ok_or_else(||self.error(Cause::Source("routed scope controls overflow")))?)
            .map_err(|e|self.error(e.into()))?;
        let coordinate=prefill.map(|(chunk,inference)|(inference,chunk.input.start/inference.prefill_chunk_positions));
        let slots=self.routed_hooks.take().ok_or_else(||self.error(Cause::Source("sparse hook table is already lent")))?;
        let mut scope=PartitionCaptureRoutedHooks{slots,first,prefill:coordinate,source:self.source.clone(),metadata:self.metadata.clone()};
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureProgramError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let result=(||{
            for index in 0..self.rows.len() {
                if !PartitionCaptureRoutedHooks::same_routing(self.source.admission(),first,index)
                    ||!self.source.admission().plan().selections[index].schedule.includes(self.context.phase,self.context.prediction){continue;}
                let Some(slot)=self.projected.get_mut(index).and_then(Option::as_mut) else{return Err(error(Cause::Source("matching sparse point lacks its original source")))};
                if matches!(slot,projected::Row::Skipped{..}){continue;}
                let projected::Row::Ready(row)=slot else{return Err(error(Cause::Source("sparse point is not ready")))};
                if !row.is_routed()||!row.source_present(){return Err(error(Cause::Source("actual provider differs from absent retained sparse source")));}
                let chunk=if let Some((chunk,inference))=prefill {
                    if row.geometry()!=Some(inference){return Err(error(Cause::Source("sparse prefill geometry differs")));}
                    let path=&self.source.admission().plan().selections[index].path;
                    let decision=frame.begin_routed_prefill_batch(index,chunk,path).map_err(|e|error(e.into()))?;
                    if decision==CapturePrefillHookDecision::Ignore{continue;}
                    if decision==CapturePrefillHookDecision::First{row.reserve_target(frame)?;}
                    chunk.input.start/inference.prefill_chunk_positions
                }else{
                    if row.geometry().is_some(){return Err(error(Cause::Source("prefill sparse row cannot issue decode")));}
                    if matches!(frame.records()[index].outcome,CaptureOutcome::Skipped{..}){continue;}
                    if !matches!(frame.records()[index].outcome,CaptureOutcome::Missing){return Err(error(Cause::Source("sparse invocation record is already consumed")));}
                    0
                };
                if !matches!(scope.slots[index],Slot::Unused){return Err(error(Cause::Source("sparse hook slot was not returned")));}
                scope.slots[index]=Slot::Active(row.take_local(chunk,epoch)?);
            }Ok(())
        })();match result{Ok(())=>Ok(scope),Err(cause)=>Err(scope.failed(cause))}
    }
    pub(super) fn restore_routed_hooks(&mut self,mut scope:PartitionCaptureRoutedHooks,frame:&mut ScheduledCaptureStep<'_>,
        prefill:Option<(&crate::prefill::PrefillChunk,InferenceGeometry)>)->Result<(),PartitionCaptureProgramError> {
        self.metadata.reserve_metadata(PartitionCaptureRoutedHooks::table_control_bytes().ok_or_else(||self.error(Cause::Source("routed return controls overflow")))?)
            .map_err(|e|self.error(e.into()))?;
        let coordinate=prefill.map(|(chunk,inference)|(inference,chunk.input.start/inference.prefill_chunk_positions));
        if self.routed_hooks.is_some()||!self.source.same_storage(&scope.source)||!self.metadata.same_account(&scope.metadata)
            ||self.rows.len()!=scope.slots.len()||scope.prefill!=coordinate {
            return Err(scope.reject("sparse hook scope belongs to another original program"));
        }
        let result=(||{
            for index in 0..scope.slots.len(){
                let Slot::Active(hook)=&mut scope.slots[index] else{continue};
                if !PartitionCaptureRoutedHooks::same_routing(self.source.admission(),scope.first,index){return Err(self.error(Cause::Source("sparse hook routing changed")));}
                if let Some(hook)=hook.take(){
                    let Some(projected::Row::Ready(row))=self.projected.get_mut(index).and_then(Option::as_mut) else{return Err(hook.reject())};
                    row.return_local(hook)?;
                }
                if let Some((_,inference))=prefill {
                    let plan=CaptureRoutedPrefillPlan::prepare(self.source.admission(),index,inference)
                        .map_err(|e|self.error(Cause::Progress(e.into())))?;
                    let fragment=plan.fragment(coordinate.expect("prefill coordinate").1).map_err(|e|self.error(Cause::Progress(e.into())))?;
                    frame.finish_partition_routed_prefill_hook(index,&fragment).map_err(|e|self.error(e.into()))?;
                }
                scope.slots[index]=Slot::Unused;
            }Ok(())
        })();if let Err(cause)=result{return Err(scope.failed(cause));}
        self.routed_hooks=Some(scope.slots);Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
