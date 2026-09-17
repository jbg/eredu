//! Sparse batches borrow the same detached original receipt and fragment bank.
use super::*;
use super::observe::{LocalCause,PartitionLocalCaptureFailure};
use crate::capture::{FundedCaptureError,ScheduledCaptureBackend};
use crate::capture::partition::PartitionCaptureProgramError;
use eredu_core::{InferenceGeometry,capture::PartitionRoutedUnitCaptureLayout};

#[derive(Debug)]
pub(super) struct Invocation {
    chunk:Option<u64>, last:bool, native_rows:u64, source_tokens:u64, completed:u64,
}
impl PartitionLocalCaptureHook {
    fn routed_result<E:std::error::Error+Send+Sync+'static>(self,result:Result<(),LocalCause<E>>)
        ->Result<Self,PartitionCaptureProgramError> {
        match result {Ok(())=>Ok(self),Err(cause)=>{
            let source=self.source().clone();let metadata=self.funding().clone();
            Err(PartitionCaptureProgramError::local(PartitionLocalCaptureFailure{cause,_hook:self},source,metadata))
        }}
    }
    /// The input validation precedes destination allocation, but input dtype is
    /// not treated as a witness for the selected grouped result. Each actual
    /// result batch validates its own dtype against the retained native source.
    pub(crate) fn begin_routed_program<T,E:std::error::Error+Send+Sync+'static>(mut self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,
        invocation:&crate::RoutedUnitInvocation<'_,T>,prefill:Option<(InferenceGeometry,u64)>)
        ->Result<Self,PartitionCaptureProgramError> {
        let result=(||{
            self.metadata.reserve_metadata(Self::routed_control_bytes::<T,E>().ok_or(LocalCause::Identity)?)?;
            if self.routed.is_some(){return Err(LocalCause::Identity);}
            let receipt=&self.receipt;let rank=self.bank.allowance.local_rank();
            let owner=self.routed_source.as_ref().ok_or(LocalCause::Identity)?;
            if owner.producer!=rank{return Err(LocalCause::Identity);}
            let count=receipt.producer(rank).map_or(0,|p|p.fragments().len());
            let global=CaptureRoutedUnitsGeometry::prepare(self.source().admission(),receipt.context().selection_index,
                receipt.context().phase,receipt.context().prediction,receipt.context().invocation)?;
            if global.source_shape()[0] as u64!=owner.source_tokens{return Err(LocalCause::Identity);}
            let (chunk,last,source_tokens)=if let Some((inference,chunk))=prefill {
                let logical=CaptureRoutedPrefillPlan::prepare(self.source().admission(),receipt.context().selection_index,inference)?;
                let source_tokens=logical.fragment(chunk)?.source_tokens();
                (Some(chunk),chunk+1==logical.chunk_count(),source_tokens)
            }else{(None,true,owner.source_tokens)};
            let layout=PartitionRoutedUnitCaptureLayout{geometry:global.bank(),source_tokens,ownership:&owner.ownership};
            let (native_rows,_input_dtype)=backend.validate_partition_routed_invocation(invocation,&layout)?;
            if native_rows!=0&&owner.dtype.is_none(){return Err(LocalCause::Identity);}
            backend.complete_partition_source(invocation.input)?;
            for fragment in 0..count {
                let (dtype,estimate)=self.bank.allowance.fragment_source(rank,fragment)
                    .map(|(dtype,estimate)|(dtype.clone(),estimate)).ok_or(LocalCause::Identity)?;
                if chunk.is_none() || chunk==Some(0) {
                    let mut loan=if chunk.is_some(){self.bank.begin_local_routed_prefill(receipt,fragment,&dtype,estimate)?}
                        else{self.bank.begin_local_routed(receipt,fragment,&dtype,estimate,native_rows)?};
                    loan.quota_mut().reserve_quota(estimate.capture)?;
                }
                if let Some(chunk)=chunk {self.bank.begin_local_routed_invocation(receipt,fragment,chunk,native_rows)?;}
            }
            self.routed=Some(Invocation{chunk,last,native_rows,source_tokens,completed:0});Ok(())
        })();self.routed_result(result)
    }
    /// No vector of batch outputs is retained. Every callback consumes its
    /// existing native source and seals the short writer on the original target.
    pub(crate) fn observe_routed_program<T,E:std::error::Error+Send+Sync+'static>(mut self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,batch:&crate::RoutedUnitBatch<'_,T>)
        ->Result<Self,PartitionCaptureProgramError> {
        let result=(||{
            self.metadata.reserve_metadata(Self::routed_control_bytes::<T,E>().ok_or(LocalCause::Identity)?)?;
            let state=self.routed.as_ref().ok_or(LocalCause::Identity)?;
            let owner=self.routed_source.as_ref().ok_or(LocalCause::Identity)?;
            let expected=owner.dtype.as_ref().ok_or(LocalCause::Identity)?;
            let receipt=&self.receipt;let rank=self.bank.allowance.local_rank();
            let count=receipt.producer(rank).map_or(0,|p|p.fragments().len());
            let source=batch.partition_capture_source()?;
            let global=CaptureRoutedUnitsGeometry::prepare(self.source().admission(),receipt.context().selection_index,
                receipt.context().phase,receipt.context().prediction,receipt.context().invocation)?;
            let layout=PartitionRoutedUnitCaptureLayout{geometry:global.bank(),source_tokens:state.source_tokens,ownership:&owner.ownership};
            let (dtype,batch_rows)=backend.validate_partition_routed_batch_source(&source,&layout,state.native_rows)?;
            if &dtype!=expected{return Err(LocalCause::Identity);}
            let completed=PartitionRoutedUnitCaptureLayout::advance_native_rows(state.native_rows,state.completed,source.source.token_offset,batch_rows)?;
            if count==0 {
                // A real replica/empty-overlap callback still settles every
                // actual descriptor under the existing ordinary root owner.
                for value in [source.source.values,source.source.token_indices,source.source.selection_indices,
                    source.source.coefficients,source.source.source_groups] {backend.complete_partition_source(value)?;}
            }
            for fragment in 0..count {
                let (dtype,estimate)=self.bank.allowance.fragment_source(rank,fragment)
                    .map(|(dtype,estimate)|(dtype.clone(),estimate)).ok_or(LocalCause::Identity)?;
                let writer=self.bank.take_local_routed_writer(receipt,fragment)?;
                if writer.native_rows()!=state.native_rows || writer.invocation_source_tokens()!=state.source_tokens {
                    return Err(LocalCause::Identity);
                }
                let request=writer.request();
                let actual=backend.validate_partition_routed_source(&source,&request,state.source_tokens,state.native_rows)?;
                let usage=backend.estimate_partition_routed(&request)?;
                if actual!=dtype || estimate!= (PartitionCaptureNativeEstimate{capture:usage,generated_creation_bytes:0}) {
                    return Err(LocalCause::Identity);
                }
                backend.transform_partition_routed_batch(&source,writer)?;
            }
            self.routed.as_mut().expect("validated active sparse invocation").completed=completed;
            Ok(())
        })();self.routed_result(result)
    }
    /// Exact native range checks remain in the shared sparse Host writer. Final
    /// prefill seal requires every canonical input invocation, including idle EP.
    pub(crate) fn finish_routed_program<T,E:std::error::Error+Send+Sync+'static>(mut self,success:bool)
        ->Result<Self,PartitionCaptureProgramError> {
        let result=(||{
            self.metadata.reserve_metadata(Self::routed_control_bytes::<T,E>().ok_or(LocalCause::Identity)?)?;
            if !success{return Err(LocalCause::Identity);}
            let state=self.routed.as_ref().ok_or(LocalCause::Identity)?;
            if state.completed!=state.native_rows{return Err(LocalCause::Identity);}
            let receipt=&self.receipt;let rank=self.bank.allowance.local_rank();
            let count=receipt.producer(rank).map_or(0,|p|p.fragments().len());
            for fragment in 0..count {
                if let Some(chunk)=state.chunk {self.bank.finish_local_routed_invocation(receipt,fragment,chunk)?;}
                if state.last {self.bank.finish_local_routed(receipt,fragment)?;}
            }
            self.routed=None;Ok(())
        })();self.routed_result::<E>(result)
    }
    pub(crate) fn routed_control_bytes<T,E:std::error::Error+Send+Sync+'static>()->Option<usize> {
        let parts=[size_of::<[&T;5]>(),size_of::<crate::capture::partition::PreparedPartitionRoutedLocalSource>(),size_of::<CaptureRoutedUnitsGeometry<'_>>(),size_of::<Result<(TensorDtype,u64),FundedCaptureError<E>>>(),crate::capture::partition::PartitionCaptureLocalHook::control_bytes()?,
            CapturePartitionRoutedHostPlan::control_bytes()?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionLocalCaptureFailure<E>>()?,
            size_of::<PartitionLocalCaptureHook>()*2,size_of::<Invocation>(),size_of::<Option<Invocation>>(),
            size_of::<PartitionLocalCaptureFailure<E>>(),size_of::<LocalCause<E>>(),size_of::<FundedCaptureError<E>>(),
            size_of::<Result<PartitionLocalCaptureHook,PartitionCaptureProgramError>>(),size_of::<Result<(),LocalCause<E>>>(),
            size_of::<PartitionCaptureProgramError>(),size_of::<PartitionRoutedUnitCaptureLayout<'_>>(),
            size_of::<PartitionRoutedUnitCaptureRequest<'_>>(),size_of::<PartitionRoutedUnitCaptureSource<'_,T>>(),
            size_of::<crate::RoutedUnitInvocation<'_,T>>(),size_of::<crate::RoutedUnitBatch<'_,T>>(),
            size_of::<CaptureRoutedPrefillPlan<'_>>(),size_of::<CaptureRoutedPrefillFragment<'_,'_>>(),
            size_of::<CapturePartitionRoutedWriter<'_,'_>>(),size_of::<PreparedPartitionFragmentLoan>(),
            size_of::<PartitionCaptureNativeEstimate>(),size_of::<TensorDtype>()*2,size_of::<CaptureUsage>(),
            size_of::<Result<(u64,TensorDtype),FundedCaptureError<E>>>(),
            size_of::<Result<TensorDtype,FundedCaptureError<E>>>(),size_of::<Result<CaptureUsage,CaptureError>>(),
            size_of::<Result<CapturePartitionRoutedWriter<'_,'_>,PartitionFragmentDestinationError>>(),
            size_of::<Option<(InferenceGeometry,u64)>>(),size_of::<(Option<u64>,bool,u64)>(),
            size_of::<(&mut PartitionLocalCaptureHook,&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&crate::RoutedUnitInvocation<'_,T>,Option<(InferenceGeometry,u64)>)>(),
            size_of::<(&mut PartitionLocalCaptureHook,&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&crate::RoutedUnitBatch<'_,T>)>(),
            size_of::<SharedCapturePlan>(),size_of::<WorkspaceMetadataFunding>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
