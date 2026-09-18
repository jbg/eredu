//! Original fragment destinations through the existing all-rank receipt worker.
use super::*;
use crate::capture::partition::{PartitionCaptureDecoder, PartitionCaptureExchange,
    PartitionCaptureExchangeError, PartitionCapturePayload, PartitionCaptureTransport,
    PreparedPartitionCaptureEvidence, PartitionCaptureEvidenceError};
use eredu_core::{BackendFailure, Completion};

/// Actual cold producer witness, including genuine empty shards. Nonempty dtype
/// must match the native allowance; host F32 payloads supply no source precision.
#[derive(Debug)]
pub(crate) struct PartitionCaptureRankSource {
    pub(crate) producer:usize,
    pub(crate) dtype:Option<TensorDtype>,
}
#[derive(Debug,thiserror::Error)]
enum Cause {
    #[error("contiguous delivery source, rank, precision or destination differs")] Source,
    #[error(transparent)] Funding(#[from] HostMetadataFundingError),
    #[error(transparent)] Destination(#[from] eredu_nn::Error),
    #[error(transparent)] Memory(#[from] WorkingMemoryError),
    #[error(transparent)] Host(#[from] CaptureRunHostError),
    #[error(transparent)] Capture(#[from] CaptureError),
    #[error(transparent)] Native(#[from] PartitionCaptureFragmentAllowanceError),
    #[error(transparent)] Evidence(#[from] PartitionCaptureEvidenceError),
    #[error(transparent)] Encode(#[from] PartitionFragmentEncodingError),
    #[error(transparent)] Receive(#[from] PartitionFragmentReceiveError),
    #[error(transparent)] Assemble(#[from] PartitionFragmentAssemblyError),
    #[error(transparent)] Exchange(#[from] PartitionCaptureExchangeError),
}
/// Failure never returns a consumed final claim or once-only protocol owner.
#[derive(Debug,thiserror::Error)]
#[error("contiguous partition delivery: {cause}")]
pub(crate) struct PartitionFragmentDeliveryError {
    #[source] cause:Cause,
    _bank:Option<PreparedPartitionFragmentDestinations>,
    _source:SharedCapturePlan,
    _fragment_host:CaptureTensorCustody,
    _final_host:Option<CaptureTensorCustody>,
    _metadata:HostMetadataFunding,
}
/// Released only after the existing final delivery vote succeeds. The enclosing
/// frame still authenticates its final claim; no prefill progress is invented.
#[derive(Debug)]
pub(crate) struct PartitionFragmentDelivered {
    pub(crate) value:PartitionFragmentValue,
    pub(crate) evidence:PreparedPartitionCaptureEvidence,
    pub(crate) charged:CaptureUsage,
    pub(crate) dtype:TensorDtype,
}
pub(crate) struct PreparedPartitionFragmentDelivery<'t,T:PartitionCaptureTransport> {
    exchange:PartitionCaptureExchange<'t,T>,bank:PreparedPartitionFragmentDestinations,
    ranks:Vec<PartitionCaptureRankSource>,evidence:PreparedPartitionCaptureEvidence,
    dtype:TensorDtype,metadata:HostMetadataFunding,
}
impl<T:PartitionCaptureTransport> fmt::Debug for PreparedPartitionFragmentDelivery<'_,T> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("PreparedPartitionFragmentDelivery").field("ranks",&self.ranks).finish_non_exhaustive()
    }
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionFragmentDelivery<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    /// Prepare before forward using the existing common quota. Native operation
    /// and transport qualification remain independently required.
    pub(crate) fn prepare(transport:&'t T,receipt:PartitionCaptureReceiptPlan,
        mut bank:PreparedPartitionFragmentDestinations,ranks:&[PartitionCaptureRankSource],
        metadata:&HostMetadataFunding)->Result<Self,PartitionFragmentDeliveryError> {
        match Self::prepare_worker(transport,receipt,&mut bank,ranks,metadata) {
            Ok((exchange,ranks,evidence,dtype))=>Ok(Self{exchange,bank,ranks,evidence,dtype,metadata:metadata.clone()}),
            Err(cause)=>Err(PartitionFragmentDeliveryError{cause,_source:bank.source.clone(),
                _fragment_host:bank.custody.share_scheduled(),_final_host:None,_bank:Some(bank),_metadata:metadata.clone()}),
        }
    }
    fn prepare_worker(transport:&'t T,receipt:PartitionCaptureReceiptPlan,
        bank:&mut PreparedPartitionFragmentDestinations,ranks:&[PartitionCaptureRankSource],metadata:&HostMetadataFunding)
        ->Result<(PartitionCaptureExchange<'t,T>,Vec<PartitionCaptureRankSource>,PreparedPartitionCaptureEvidence,TensorDtype),Cause> {
        let producers=receipt.producers();
        let source_rows=receipt.producers().zip(ranks);
        let parts=[size_of::<Self>()*2,size_of::<PartitionFragmentDeliveryError>()*3,size_of::<Cause>()*2,
            size_of::<Result<Self,PartitionFragmentDeliveryError>>(),size_of::<PartitionFragmentDelivered>()*2,
            size_of::<Result<PartitionFragmentDelivered,PartitionFragmentDeliveryError>>(),
            size_of::<(PartitionCaptureExchange<'t,T>,Vec<PartitionCaptureRankSource>,PreparedPartitionCaptureEvidence,TensorDtype)>(),
            size_of::<Result<(PartitionCaptureExchange<'t,T>,Vec<PartitionCaptureRankSource>,PreparedPartitionCaptureEvidence,TensorDtype),Cause>>(),
            size_of::<(&T,PartitionCaptureReceiptPlan,PreparedPartitionFragmentDestinations,&[PartitionCaptureRankSource],&HostMetadataFunding)>(),
            size_of::<(&T,&mut PreparedPartitionFragmentDestinations,&[PartitionCaptureRankSource],&HostMetadataFunding)>(),
            size_of::<PartitionCaptureRankSource>()*2,size_of::<Option<TensorDtype>>()*2,
            size_of::<CaptureTensorCustody>()*3,size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>(),
            size_of::<Option<PreparedPartitionFragmentDestinations>>(),size_of::<CaptureUsage>()*3,
            size_of_val(&producers),size_of_val(&source_rows),size_of::<std::slice::Iter<'_,PartitionCaptureRankSource>>(),
            size_of::<(usize,bool)>(),size_of::<(&mut Decoder<'_,'_,'_>,&PartitionCaptureReceiptPlan,usize)>(),
            size_of::<PartitionFragmentDestination<'_,'_>>(),size_of::<Result<(),Cause>>(),
            size_of::<(Cause,Option<PreparedPartitionFragmentDestinations>,&SharedCapturePlan,&CaptureTensorCustody,Option<&CaptureTensorCustody>,&HostMetadataFunding)>(),
            size_of::<(&mut PreparedPartitionFragmentDestinations,&Vec<PartitionCaptureRankSource>,&PartitionCaptureReceiptPlan,&mut usize,&SharedCapturePlan,&CaptureTensorCustody,&CaptureTensorCustody,&HostMetadataFunding)>(),
            size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,&Vec<PartitionCaptureRankSource>,&HostMetadataFunding)>(),
            // Named owning source-error boxes, before any fallible forward work.
            BackendFailure::source_retention_peak_bytes::<PartitionFragmentDeliveryError>().ok_or(Cause::Source)?.checked_mul(2).ok_or(Cause::Source)?,
            PartitionCaptureExchange::<T>::decoder_control_bytes::<Decoder<'_,'_,'_>>().ok_or(Cause::Source)?,
            if bank.prefill.is_some(){Self::prefill_control_bytes(&receipt).ok_or(Cause::Source)?}
                else{Self::invocation_control_bytes().ok_or(Cause::Source)?},
            hook::control_bytes::<T>().ok_or(Cause::Source)?,
        ];
        metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or(Cause::Source)?)?;
        bank.custody.validate()?;
        if !bank.matches(&receipt) || bank.allowance.local_rank()!=transport.capture_rank()
            || receipt.world_size()!=transport.participant_count() || ranks.len()!=producers.count() {return Err(Cause::Source);}
        let mut dtype=None;let mut any_fragment=false;
        for ((producer,projection),rank) in source_rows {
            let routed=receipt.routed_producer(producer).is_some();
            if rank.producer!=producer || !routed&&projection.fragments().len()>1
                || !matches!(rank.dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16))
                    && !(routed&&rank.dtype==Some(TensorDtype::F64)) {return Err(Cause::Source);}
            for fragment in 0..projection.fragments().len() {
                let (actual,_)=bank.allowance.fragment_source(producer,fragment).ok_or(Cause::Source)?;
                if (!routed||rank.dtype.is_some())&&rank.dtype.as_ref()!=Some(actual)
                    || dtype.as_ref().is_some_and(|first|first!=actual){return Err(Cause::Source);}
                if rank.dtype.is_some(){dtype=Some(actual.clone());}any_fragment=true;
            }
        }
        // Empty selections preserve their actual common source precision.
        if !any_fragment {for rank in ranks {
            if let Some(first)=&dtype {if rank.dtype.as_ref()!=Some(first){return Err(Cause::Source);}}
            else {dtype=Some(rank.dtype.clone().ok_or(Cause::Source)?);}
        }}
        let dtype=dtype.ok_or(Cause::Source)?;
        for (producer,projection) in receipt.producers(){for fragment in 0..projection.fragments().len(){
            if bank.allowance.fragment_source(producer,fragment).map(|(actual,_)|actual)!=Some(&dtype){return Err(Cause::Source);}
        }}
        let mut owned=metadata.metadata_vec(ranks.len())?;
        for rank in ranks {owned.push(PartitionCaptureRankSource{producer:rank.producer,dtype:rank.dtype.clone()});}
        let evidence=PreparedPartitionCaptureEvidence::prepare_contiguous(&receipt,&mut bank.allowance,metadata)?;
        bank.allowance.quota_mut().reserve_quota(receipt.delivery_table_usage()?)?;
        let local=transport.capture_rank();
        if let Some(projection)=receipt.producer(local) {
            let overhead=bank.allowance.fragment_record_overhead(&receipt,local).ok_or(Cause::Source)?;
            bank.allowance.quota_mut().reserve_quota(overhead.checked_mul(projection.fragments().len() as u64)?)?;
        }
        let exchange=PartitionCaptureExchange::admit(transport,receipt,bank.allowance.quota_mut())?;
        Ok((exchange,owned,evidence,dtype))
    }
    fn prefill_control_bytes(receipt:&PartitionCaptureReceiptPlan)->Option<usize> {
        let raw=receipt.combination()==PartitionCaptureCombination::SumF64ToF32 || matches!(
            receipt.shared_plan_source().admission().plan().selections[receipt.context().selection_index].transform,
            CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..});
        let payload=if receipt.producers().any(|(rank,_)|receipt.routed_producer(rank).is_some()) {
            size_of::<CaptureRoutedClaim<'_,'_>>().checked_add(size_of::<ClaimedAssembledRoutedUnits>())?
        }else if raw {
            size_of::<Result<crate::working_memory::CapturePrefillFragmentClaim<'_,'_,'_,'_>,PartitionFragmentDestinationError>>()
                .checked_add(size_of::<(&mut Self,usize,&CapturePrefillFragment<'_,'_>)>())?
        }else{
            size_of::<Result<PartitionFragmentDestination<'_,'_>,PartitionFragmentDestinationError>>()
                .checked_add(size_of::<(&mut Self,usize,&CapturePrefillTransformFragment<'_,'_>)>())?
                .checked_add(size_of::<(&mut Self,usize,&CapturePrefillTransformFragment<'_,'_>,PartitionFragmentValue)>())?
        };
        let parts=[payload,size_of::<(&mut Self,&mut ScheduledCaptureStep<'_>)>(),size_of::<(Self,&mut ScheduledCaptureStep<'_>)>(),
            size_of::<(&mut Self,usize,&TensorDtype,PartitionCaptureNativeEstimate)>(),size_of::<(&mut Self,usize,u64)>(),
            size_of::<(&mut Self,usize)>(),size_of::<Result<(),PartitionFragmentDestinationError>>(),
            size_of::<Result<(),PartitionFragmentDeliveryError>>(),size_of::<Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError>>(),
            size_of::<PartitionFragmentDelivered>(),size_of::<PartitionFragmentDeliveryError>(),size_of::<PartitionFragmentDestinationError>(),
            size_of::<SharedCapturePlan>()*2,size_of::<CaptureTensorCustody>()*3,size_of::<HostMetadataFunding>(),
            size_of::<(&SharedCapturePlan,&CaptureTensorCustody,&CaptureTensorCustody,&HostMetadataFunding)>(),
            size_of::<crate::capture::partition::PreparedPartitionAssemblyCharge<'_>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Bind the already globally charged assembly to this frame's real first
    /// hook. Native credits remain in the separate local fragment allowance.
    pub(crate) fn reserve_prefill_target(&mut self,frame:&mut ScheduledCaptureStep<'_>)
        ->Result<(),PartitionFragmentDestinationError> {
        let receipt=self.exchange.receipt_plan();
        let source=self.bank.source.clone();let custody=self.bank.custody.share_scheduled();
        let error=|cause|PartitionFragmentDestinationError{cause,_value:None,_source:source.clone(),_custody:Some(custody.share_scheduled())};
        let charge=self.bank.allowance.take_assembly_charge(receipt).map_err(|cause|error(cause.into()))?;
        frame.reserve_assembled_prefill_hook(receipt.context().selection_index,self.dtype.clone(),charge)
            .map_err(|cause|error(cause.into()))
    }
    pub(crate) fn begin_local_prefill(&mut self,fragment:usize,dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate)
        ->Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError> {
        self.bank.begin_local_prefill(self.exchange.receipt_plan(),fragment,dtype,actual)
    }
    pub(crate) fn take_local_prefill_chunk<'c,'f,'p,'a>(&'c mut self,fragment:usize,chunk:&'f CapturePrefillFragment<'p,'a>)
        ->Result<crate::working_memory::CapturePrefillFragmentClaim<'c,'f,'p,'a>,PartitionFragmentDestinationError> {
        self.bank.take_local_prefill_chunk(self.exchange.receipt_plan(),fragment,chunk)
    }
    pub(crate) fn take_local_prefill_reduction<'c>(&'c mut self,fragment:usize,chunk:&CapturePrefillTransformFragment<'_,'_>)
        ->Result<PartitionFragmentDestination<'c,'c>,PartitionFragmentDestinationError> {
        self.bank.take_local_prefill_reduction(self.exchange.receipt_plan(),fragment,chunk)
    }
    pub(crate) fn record_local_prefill_reduction(&mut self,fragment:usize,chunk:&CapturePrefillTransformFragment<'_,'_>,value:PartitionFragmentValue)
        ->Result<(),PartitionFragmentDestinationError> {
        self.bank.record_local_prefill_reduction(self.exchange.receipt_plan(),fragment,chunk,value)
    }
    pub(crate) fn complete_local_prefill_chunk(&mut self,fragment:usize,chunk:u64)
        ->Result<(),PartitionFragmentDestinationError> {
        self.bank.complete_local_prefill_chunk(self.exchange.receipt_plan(),fragment,chunk)
    }
    pub(crate) fn finish_local_prefill(&mut self,fragment:usize)->Result<(),PartitionFragmentDestinationError> {
        self.bank.finish_local_prefill(self.exchange.receipt_plan(),fragment)
    }
    pub(crate) fn finish_local_invocation(&self,fragment:usize)->Result<(),PartitionFragmentDestinationError> {
        let receipt=self.exchange.receipt_plan();let local=self.bank.allowance.local_rank();
        if self.bank.prefill.is_some()||!self.bank.matches(receipt)||self.bank.value(local,fragment).is_none(){
            return Err(self.bank.error(FragmentCause::Source,None));
        }
        Ok(())
    }
    /// Final assembly and delivery vote precede attachment to the scheduled
    /// frame. Original logical chunk progression must already be finished.
    pub(crate) fn deliver_into_prefill_frame(self,frame:&mut ScheduledCaptureStep<'_>)
        ->Result<(),PartitionFragmentDeliveryError> {
        let source=self.bank.source.clone();let fragment_host=self.bank.custody.share_scheduled();
        let final_host=frame.claim.custody.share_scheduled();let metadata=self.metadata.clone();
        let error=|cause,bank|PartitionFragmentDeliveryError{cause,_bank:bank,_source:source.clone(),
            _fragment_host:fragment_host.share_scheduled(),_final_host:Some(final_host.share_scheduled()),_metadata:metadata.clone()};
        let context=self.exchange.receipt_plan().context();
        if !source.same_storage(frame.claim.source) || context.phase!=frame.claim.phase
            || context.prediction!=frame.claim.prediction || frame.claim.invocation.is_some() || frame.claim.window.is_some() {
            return Err(error(Cause::Source,Some(self.bank)));
        }
        let destination=match frame.take_assembled_prefill_destination(context.selection_index) {
            Ok(destination)=>destination,Err(cause)=>return Err(error(cause.into(),Some(self.bank))),
        };
        let result=self.deliver(destination)?;
        frame.record_assembled_prefill(result).map_err(|cause|error(cause.into(),None))
    }
    /// Same exchange/decoder/assembler, with a single ordinary global target.
    /// Its native terms and final record were already charged by the original
    /// fragment allowance; final vote success precedes frame attachment.
    pub(crate) fn deliver_into_invocation_frame(mut self,frame:&mut ScheduledCaptureStep<'_>)
        ->Result<(),PartitionFragmentDeliveryError> {
        let source=self.bank.source.clone();let fragment_host=self.bank.custody.share_scheduled();
        let final_host=frame.claim.custody.share_scheduled();let metadata=self.metadata.clone();
        let error=|cause,bank|PartitionFragmentDeliveryError{cause,_bank:bank,_source:source.clone(),
            _fragment_host:fragment_host.share_scheduled(),_final_host:Some(final_host.share_scheduled()),_metadata:metadata.clone()};
        let (destination,target)=match self.invocation_target(frame) {
            Ok(value)=>value,Err(cause)=>return Err(error(cause,Some(self.bank))),
        };
        let result=self.deliver(destination)?;
        target.record(frame,result).map_err(|cause|error(cause.into(),None))
    }
    fn invocation_control_bytes()->Option<usize> {
        let parts=[super::super::invocation_assembly_control_bytes()?,
            size_of::<(Self,&mut ScheduledCaptureStep<'_>)>(),size_of::<(&mut Self,&mut ScheduledCaptureStep<'_>)>(),
            size_of::<Result<(),PartitionFragmentDeliveryError>>(),size_of::<PartitionFragmentDeliveryError>(),
            size_of::<super::super::assembled::InvocationAssemblyTarget<'_>>(),
            size_of::<Result<(PartitionFragmentDestination<'_,'_>,super::super::assembled::InvocationAssemblyTarget<'_>),Cause>>(),
            size_of::<SharedCapturePlan>(),size_of::<CaptureTensorCustody>()*3,size_of::<HostMetadataFunding>(),
            size_of::<(&SharedCapturePlan,&CaptureTensorCustody,&CaptureTensorCustody,&HostMetadataFunding)>(),
            size_of::<(Cause,Option<PreparedPartitionFragmentDestinations>)>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    fn invocation_target<'a,'c>(&mut self,frame:&'c mut ScheduledCaptureStep<'a>)
        ->Result<(PartitionFragmentDestination<'a,'c>,super::super::assembled::InvocationAssemblyTarget<'a>),Cause> {
        let context=self.exchange.receipt_plan().context();
        if !self.bank.source.same_storage(frame.claim.source)||context.phase!=frame.claim.phase
            ||context.prediction!=frame.claim.prediction||frame.claim.invocation.is_some()||frame.claim.window.is_some()
            ||self.bank.prefill.is_some(){return Err(Cause::Source);}
        let index=context.selection_index;
        let charge=self.bank.allowance.take_assembly_charge(self.exchange.receipt_plan())?;
        frame.take_assembled_invocation_destination(index,self.dtype.clone(),charge).map_err(Into::into)
    }
    pub(crate) fn take_local<'c>(&'c mut self,fragment:usize,dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate)
        ->Result<NativePartitionFragmentDestination<'c,'c>,PartitionFragmentDestinationError> {
        self.bank.take_local(self.exchange.receipt_plan(),fragment,dtype,actual)
    }
    pub(crate) fn record_local(&mut self,fragment:usize,dtype:&TensorDtype,charged:CaptureUsage,value:PartitionFragmentValue)
        ->Result<(),PartitionFragmentDestinationError> {
        self.bank.record(self.exchange.receipt_plan(),self.bank.allowance.local_rank(),fragment,dtype,charged,value)
    }
    pub(crate) fn receipt_plan(&self)->&PartitionCaptureReceiptPlan {self.exchange.receipt_plan()}
    pub(crate) fn deliver(self,destination:PartitionFragmentDestination<'_,'_>)
        ->Result<PartitionFragmentDelivered,PartitionFragmentDeliveryError> {
        let Self{exchange,mut bank,ranks,evidence,dtype,metadata}=self;
        let source=bank.source.clone();let fragment_host=bank.custody.share_scheduled();
        let final_host=destination.identity().custody.share_scheduled();
        let local=(||->Result<_,Cause>{
            let receipt=exchange.receipt_plan();let local=bank.allowance.local_rank();
            if receipt.producer(local).is_none(){return Ok(None);}
            bank.allowance.quota_mut().reserve_quota(receipt.encoding_usage(local)?)?;
            let rank=ranks.iter().find(|rank|rank.producer==local).ok_or(Cause::Source)?;
            Ok(Some(bank.encode_contiguous_producer(receipt,rank.dtype.as_ref(),&metadata)?))
        })().map_err(|cause|as_exchange(cause,None,&source,&fragment_host,Some(&final_host),&metadata));
        let decoder=Decoder{bank,ranks,evidence,dtype,destination,metadata:&metadata};
        exchange.exchange_into(local,decoder).map_err(|cause|PartitionFragmentDeliveryError{
            cause:cause.into(),_bank:None,_source:source,_fragment_host:fragment_host,
            _final_host:Some(final_host),_metadata:metadata,
        })
    }
}
struct Decoder<'a,'c,'m> {
    bank:PreparedPartitionFragmentDestinations,ranks:Vec<PartitionCaptureRankSource>,
    evidence:PreparedPartitionCaptureEvidence,dtype:TensorDtype,
    destination:PartitionFragmentDestination<'a,'c>,metadata:&'m HostMetadataFunding,
}
fn as_exchange(cause:Cause,bank:Option<PreparedPartitionFragmentDestinations>,source:&SharedCapturePlan,
    fragment_host:&CaptureTensorCustody,final_host:Option<&CaptureTensorCustody>,metadata:&HostMetadataFunding)
    ->PartitionCaptureExchangeError {
    BackendFailure::from_error(PartitionFragmentDeliveryError{cause,_bank:bank,_source:source.clone(),
        _fragment_host:fragment_host.share_scheduled(),_final_host:final_host.map(CaptureTensorCustody::share_scheduled),
        _metadata:metadata.clone()}).into()
}
impl<T:PartitionCaptureTransport> PartitionCaptureDecoder<T> for Decoder<'_,'_,'_> {
    type Output=PartitionFragmentDelivered;
    fn decode(self,receipt:PartitionCaptureReceiptPlan,payload:PartitionCapturePayload<'_,T>)
        ->Result<Self::Output,PartitionCaptureExchangeError> {
        let Self{mut bank,ranks,mut evidence,dtype,destination,metadata}=self;
        let source=bank.source.clone();let fragment_host=bank.custody.share_scheduled();
        let final_host=destination.identity().custody.share_scheduled();let mut next=0usize;
        let result=payload.for_each(|producer,bytes| {
            let result=(||->Result<(),Cause>{
                let rank=ranks.get(next).ok_or(Cause::Source)?;
                if rank.producer!=producer{return Err(Cause::Source);}
                next+=1;
                if producer!=bank.allowance.local_rank() {
                    bank.allowance.quota_mut().reserve_quota(receipt.decoding_usage(bytes.len() as u64)?)?;
                    bank.decode_contiguous_producer(&receipt,producer,bytes,rank.dtype.as_ref(),metadata)?;
                }
                Ok(())
            })();
            result.map_err(|cause|as_exchange(cause,None,&source,&fragment_host,Some(&final_host),metadata))
        });
        if let Err(cause)=result {return Err(as_exchange(cause.into(),Some(bank),&source,&fragment_host,Some(&final_host),metadata));}
        if next!=ranks.len() || !bank.complete() {
            return Err(as_exchange(Cause::Source,Some(bank),&source,&fragment_host,Some(&final_host),metadata));
        }
        let provenance=(||->Result<(),Cause>{
            for slot in &bank.rows {if let Some(value)=&slot.routed_value {
                evidence.record_routed_ranges(&receipt,slot.producer,slot.fragment,&value.observation().source_token_ranges)?;
            }}
            if !evidence.matches(&receipt){return Err(Cause::Source);}Ok(())
        })();
        if let Err(cause)=provenance{return Err(as_exchange(cause,Some(bank),&source,&fragment_host,Some(&final_host),metadata));}
        let (value,charged)=bank.assemble_into(&receipt,destination,&dtype)
            .map_err(|cause|as_exchange(cause.into(),None,&source,&fragment_host,Some(&final_host),metadata))?;
        Ok(PartitionFragmentDelivered{value,evidence,charged,dtype})
    }
}

mod hook;
pub(crate) use hook::{PartitionLocalCaptureHook,PartitionCaptureHookContinuation,PartitionCaptureHookReturnError};

use eredu_nn::workspace::WorkspaceMetadataAllocation;
