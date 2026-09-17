//! One local projected destination across actual fixed-schedule prefill chunks.
use super::*;
use crate::working_memory::capture_tensor::prefill::{OwnedPrefillTensor,TargetSlot,TargetState};
use crate::working_memory::capture_run::claims::prefill::{CapturePrefillFragmentClaim,CapturePrefillHostError};
use eredu_core::{InferenceGeometry,TensorObservation,TensorObservationData};

#[derive(Debug)]
pub(super) struct Target { slot:TargetSlot, next:u64, pending:bool }

fn row_source<'a>(receipt:&'a PartitionCaptureReceiptPlan,producer:usize,fragment:usize,
    inference:InferenceGeometry)->Result<CapturePrefillRowAssembly<'a>,CaptureRunHostError>
{
    let source=receipt.shared_plan_source().ok_or(CaptureRunHostError::ReceiptMismatch)?;
    if receipt.context().phase!=CapturePhase::Prefill || receipt.context().prediction!=0
        || receipt.context().invocation.is_some() {return Err(CaptureRunHostError::ReceiptMismatch);}
    let projection=receipt.producer(producer).ok_or(CaptureRunHostError::ReceiptMismatch)?;
    let index=receipt.context().selection_index;
    let nonlinear=matches!(source.admission().plan().selections[index].transform,
        CaptureTransform::Summary|CaptureTransform::Histogram{..});
    if nonlinear && receipt.combination()==PartitionCaptureCombination::SumF64ToF32 {
        CapturePrefillRowAssembly::prepare_additive_transform_partition(source.admission(),index,inference,projection,fragment)
            .map_err(CapturePrefillHostError::from).map_err(Into::into)
    } else {
        // Disjoint reductions retain their typed partial-statistics/bin merger;
        // this raw target never silently substitutes a tensor for that source.
        CapturePrefillRowAssembly::prepare_partition(source.admission(),index,inference,projection,fragment,receipt.combination())
            .map_err(CapturePrefillHostError::from).map_err(Into::into)
    }
}
fn same_geometry(left:&CaptureTensorGeometry<'_>,right:&CaptureTensorGeometry<'_>)->bool {
    std::ptr::eq(left.admission(),right.admission()) && left.selection_index()==right.selection_index()
        && left.phase()==right.phase() && left.prediction()==right.prediction()
        && left.source_shape()==right.source_shape() && left.shape()==right.shape()
        && left.starts()==right.starts() && left.ends()==right.ends() && left.strides()==right.strides()
        && left.elements()==right.elements() && left.native_transform()==right.native_transform()
}
impl<'a> PartitionFragmentHostPlan<'a> {
    /// Retain the real fixed prefill schedule and its exact projected raw target
    /// controls in addition to the existing local/received whole-fragment P.
    /// Additive Summary/Histogram remain raw until the shared global assembler.
    pub fn prepare_prefill(receipt:&'a PartitionCaptureReceiptPlan,inference:InferenceGeometry)
        ->Result<Self,CaptureRunHostError>
    {
        let mut plan=Self::prepare(receipt)?;
        if plan.slots==0 {return Err(CaptureRunHostError::ReceiptMismatch);}
        for (producer,projection) in receipt.producers(){for fragment in 0..projection.fragments().len(){
            if receipt.routed_producer(producer).is_some() {
                routed::prefill_source(receipt,producer,fragment,inference)?;
            } else if reductions::selected(receipt) {
                reductions::source(receipt,producer,fragment,inference)?;
            } else {
                let rows=row_source(receipt,producer,fragment,inference)?;
                let FragmentHostPlan::Tensor(host)=FragmentHostPlan::prepare(receipt,producer,fragment)? else {
                    return Err(CaptureRunHostError::ReceiptMismatch);
                };
                if !same_geometry(rows.logical_geometry(),host.geometry()){return Err(CaptureRunHostError::ReceiptMismatch);}
            }
        }}
        let additive=receipt.combination()==PartitionCaptureCombination::SumF64ToF32 && matches!(
            receipt.shared_plan_source().expect("checked source").admission().plan().selections[receipt.context().selection_index].transform,
            CaptureTransform::Summary|CaptureTransform::Histogram{..});
        let routed=receipt.producers().any(|(producer,_)|receipt.routed_producer(producer).is_some());
        let controls=(if routed{routed::prefill_control_bytes()}else if reductions::selected(receipt){reductions::control_bytes(receipt)}else{control_bytes(additive)}).and_then(|n|u64::try_from(n).ok()).ok_or(WorkingMemoryError::Overflow)?;
        plan.table=plan.table.checked_add(controls).ok_or(WorkingMemoryError::Overflow)?;
        plan.table.checked_add(plan.fragments).and_then(|n|n.checked_add(plan.assembly_peak_bytes()))
            .ok_or(WorkingMemoryError::Overflow)?;
        plan.prefill=Some(inference);Ok(plan)
    }
}
impl PreparedPartitionFragmentDestinations {
    /// Spend one actual whole-fragment native allowance and initialize its local
    /// prefill state once. The returned loan is retained by the native observer;
    /// every chunk spends that same quota, with no per-chunk refill or refund.
    pub fn begin_local_prefill(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate)
        ->Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError>
    {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        if !self.matches(receipt)||self.prefill.is_none(){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        let loan=self.allowance.take_local_fragment(receipt,fragment,dtype,actual).map_err(|e|self.error(e.into(),None))?;
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Available)
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let row=&mut self.rows[index];row.state=State::Taken;
        row.prefill=Some(Target{next:0,pending:false,slot:TargetSlot{
            tensor:None,completed:None,summary:None,histogram:None,routed:None,routed_covered:0,
            state:TargetState::Claimed,dtype:Some(dtype.clone()),done:false,progression:None,
        }});
        Ok(loan)
    }
    /// Lend the existing ordinary scalar/scatter worker only after comparing its
    /// original source, full local geometry, schedule and next chunk exactly.
    /// Its stamped segment entry remains responsible for native source custody.
    pub fn take_local_prefill_chunk<'t,'f,'p,'a>(&'t mut self,receipt:&PartitionCaptureReceiptPlan,
        fragment:usize,chunk:&'f CapturePrefillFragment<'p,'a>)
        ->Result<CapturePrefillFragmentClaim<'t,'f,'p,'a>,PartitionFragmentDestinationError>
    {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        if !self.matches(receipt)||chunk.assembly().inference_geometry()!=inference{
            return Err(self.error(FragmentCause::Source,None));
        }
        let producer=self.allowance.local_rank();
        let expected=row_source(receipt,producer,fragment,inference).map_err(|e|self.error(e.into(),None))?;
        if !same_geometry(expected.logical_geometry(),chunk.assembly().logical_geometry()){
            return Err(self.error(FragmentCause::Source,None));
        }
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken
            && r.prefill.as_ref().is_some_and(|target|target.next==chunk.chunk_index()&&!target.slot.done
                && matches!(target.slot.state,TargetState::Claimed|TargetState::Active)))
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let row=&mut self.rows[index];let custody=row.custody.share_scheduled();
        let target=row.prefill.as_mut().expect("checked local prefill");
        Ok(CapturePrefillFragmentClaim::from_partition(&mut target.slot,chunk,receipt.context().selection_index,custody))
    }
    /// Advance only a finished actual host fragment, including a finished empty
    /// fragment. Source/chunk settlement and semantic hook progress are separate;
    /// this method cannot issue the global frame's deferred receipt claim.
    pub fn complete_local_prefill_chunk(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,chunk:u64)
        ->Result<(),PartitionFragmentDestinationError>
    {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken)
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let row=&mut self.rows[index];
        let valid=row.prefill.as_ref().is_some_and(|target|target.next==chunk
            && chunk<inference.input_positions.div_ceil(inference.prefill_chunk_positions)
            && !target.pending && target.slot.done && target.slot.state==TargetState::Active);
        if !valid {row.state=State::Failed;return Err(self.error(FragmentCause::Source,None));}
        let target=row.prefill.as_mut().expect("checked prefill");target.next+=1;target.slot.done=false;Ok(())
    }
    /// Seal complete local coverage into the same source-checked fragment bank.
    /// No scalar copy, nonlinear reduction, global claim or delivery vote occurs.
    pub fn finish_local_prefill(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize)
        ->Result<(),PartitionFragmentDestinationError>
    {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,None));}
        if reductions::selected(receipt){return self.finish_local_prefill_reduction(receipt,fragment);}
        let producer=self.allowance.local_rank();
        let (dtype,estimate)=self.allowance.fragment_source(producer,fragment)
            .map(|(dtype,estimate)|(dtype.clone(),estimate)).ok_or_else(||self.error(FragmentCause::Source,None))?;
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken)
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let row=&mut self.rows[index];
        let valid=row.prefill.as_ref().is_some_and(|target|target.next==inference.input_positions.div_ceil(inference.prefill_chunk_positions)
            && target.slot.state==TargetState::Active&&!target.slot.done&&target.slot.dtype.as_ref()==Some(&dtype)
            && target.slot.tensor.as_ref().is_some_and(|tensor|tensor.covered==tensor.data.len()));
        if !valid {row.state=State::Failed;return Err(self.error(FragmentCause::Source,None));}
        let tensor=row.prefill.as_mut().expect("checked target").slot.tensor.take().expect("complete target");
        let OwnedPrefillTensor{shape,data,covered:_,custody}=tensor;
        let observation=TensorObservation::new(shape,TensorObservationData::F32(data)).expect("checked fixed target shape");
        let tensor=SharedTensorObservation::retain(observation,custody);
        let identity=ReceiptIdentity{phase:receipt.context().phase,prediction:receipt.context().prediction,
            index:receipt.context().selection_index,custody:row.custody.share_scheduled()};
        row.prefill=None;
        self.record(receipt,producer,fragment,&dtype,estimate.capture,
            PartitionFragmentValue::Tensor(ClaimedCaptureTensor{tensor,identity}))
    }
}
fn control_bytes(additive:bool)->Option<usize> {
    let parts=[size_of::<Target>()*2,size_of::<TargetSlot>(),size_of::<InferenceGeometry>()*2,
        size_of::<CapturePrefillRowAssembly<'_>>()*2,size_of::<CapturePrefillFragment<'_,'_>>(),
        size_of::<CapturePrefillFragmentClaim<'_,'_,'_,'_>>()*2,
        size_of::<crate::working_memory::CapturePrefillFragmentWriter<'_,'_,'_,'_>>(),
        size_of::<Result<CapturePrefillFragmentClaim<'_,'_,'_,'_>,PartitionFragmentDestinationError>>(),
        size_of::<Result<CapturePrefillRowAssembly<'_>,CaptureRunHostError>>(),
        size_of::<Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError>>(),
        size_of::<PreparedPartitionFragmentLoan>()*2,size_of::<OwnedPrefillTensor>()*2,
        size_of::<Result<OwnedPrefillTensor,WorkingMemoryError>>(),
        size_of::<(&CaptureTensorGeometry<'_>,CaptureTensorCustody)>(),size_of::<TensorObservation>()*2,
        size_of::<ReceiptIdentity>(),size_of::<CaptureTensorCustody>()*2,size_of::<TensorDtype>(),
        size_of::<CaptureUsage>(),size_of::<Option<&mut Target>>(),size_of::<Option<usize>>(),
        size_of::<(usize,usize,u64,bool)>(),
        size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,&TensorDtype,PartitionCaptureNativeEstimate)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,&CapturePrefillFragment<'_,'_>)>(),
        if additive {CapturePrefillRowAssembly::additive_partition_preparation_control_bytes()?}
            else {CapturePrefillRowAssembly::partition_preparation_control_bytes()?},
    ];parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

mod reductions;
