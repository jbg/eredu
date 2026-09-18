//! Sparse local fragments retain one actual Host destination between callbacks.
use super::*;
use eredu_core::InferenceGeometry;

pub(super) fn prefill_source(receipt:&PartitionCaptureReceiptPlan,producer:usize,fragment:usize,
    inference:InferenceGeometry)->Result<(),CaptureRunHostError> {
    if receipt.context().phase!=CapturePhase::Prefill || receipt.context().prediction!=0
        || receipt.context().invocation.is_some() {return Err(CaptureRunHostError::ReceiptMismatch);}
    let source=receipt.shared_plan_source();
    let logical=CaptureRoutedPrefillPlan::prepare(source.admission(),receipt.context().selection_index,inference)
        .map_err(CapturePrefillHostError::from)?;
    let host=CapturePartitionRoutedHostPlan::prepare(receipt,producer,fragment)?;
    if host.geometry().source_shape()!=logical.geometry().source_shape()
        ||host.geometry().bank()!=logical.geometry().bank() {return Err(CaptureRunHostError::ReceiptMismatch);}
    Ok(())
}
impl PreparedPartitionFragmentDestinations {
    /// Spend the retained native allowance once and allocate its exact sparse
    /// fragment. The actual total native extent is distinct from logical tokens.
    /// Subsequent callbacks borrow this same target and cannot refill its quota.
    pub fn begin_local_routed(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate,actual_native_rows:u64)
        ->Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError> {
        self.begin_routed(receipt,fragment,dtype,actual,Some(actual_native_rows))
    }
    /// Initialize one logical prefill fragment before its first actual provider
    /// invocation. Each canonical chunk later supplies its own exact native rows.
    pub fn begin_local_routed_prefill(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate)
        ->Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError> {
        self.begin_routed(receipt,fragment,dtype,actual,None)
    }
    fn begin_routed(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        dtype:&TensorDtype,actual:PartitionCaptureNativeEstimate,actual_native_rows:Option<u64>)
        ->Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError> {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        if !self.matches(receipt)||self.prefill.is_some()!=actual_native_rows.is_none(){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        let plan=CapturePartitionRoutedHostPlan::prepare(receipt,producer,fragment)
            .map_err(|e|self.error(e.into(),None))?;
        if let Some(inference)=self.prefill {
            prefill_source(receipt,producer,fragment,inference).map_err(|e|self.error(e.into(),None))?;
        }
        let loan=self.allowance.take_local_fragment(receipt,fragment,dtype,actual)
            .map_err(|e|self.error(e.into(),None))?;
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Available)
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let row=&mut self.rows[index];row.state=State::Taken;
        let identity=ReceiptIdentity{phase:receipt.context().phase,prediction:receipt.context().prediction,
            index:receipt.context().selection_index,custody:row.custody.share_scheduled()};
        let claim=CapturePartitionRoutedClaim::from_plan(plan,identity);
        match match actual_native_rows{Some(rows)=>claim.prepare(rows),None=>claim.prepare_accumulating()} {
            Ok(target)=>self.rows[index].routed=Some(target),
            Err(cause)=>{self.rows[index].state=State::Failed;return Err(self.error(cause.into(),None));}
        }
        Ok(loan)
    }
    /// Bind the next canonical prefill invocation's real received row count.
    /// Counts accumulate under the original ownership bound, never as a guessed
    /// maximum. A second begin cannot replace an unfinished invocation.
    pub fn begin_local_routed_invocation(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        chunk:u64,actual_native_rows:u64)->Result<(),PartitionFragmentDestinationError> {
        self.routed_invocation(receipt,fragment,chunk,Some(actual_native_rows))
    }
    /// Finish the current invocation only after its exact native row coverage.
    pub fn finish_local_routed_invocation(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        chunk:u64)->Result<(),PartitionFragmentDestinationError> {
        self.routed_invocation(receipt,fragment,chunk,None)
    }
    fn routed_invocation(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        chunk:u64,rows:Option<u64>)->Result<(),PartitionFragmentDestinationError> {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        if !self.matches(receipt)||chunk>=inference.input_positions.div_ceil(inference.prefill_chunk_positions){
            return Err(self.error(FragmentCause::Source,None));
        }
        let producer=self.allowance.local_rank();
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken
            &&r.routed_invocations==chunk&&r.routed_pending==rows.is_none()&&r.routed.is_some())
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let logical=CaptureRoutedPrefillPlan::prepare(self.source.admission(),receipt.context().selection_index,inference)
            .map_err(|e|self.error(FragmentCause::Host(CapturePrefillHostError::from(e).into()),None))?;
        let token_window=logical.fragment(chunk).map_err(|e|self.error(FragmentCause::Host(CapturePrefillHostError::from(e).into()),None))?.token_window();
        if let Some(rows)=rows {
            let ownership=receipt.routed_producer(producer).ok_or_else(||self.error(FragmentCause::Source,None))?;
            let maximum=ownership.maximum_source_rows_checked(token_window.source_tokens(),logical.geometry().bank().routes_per_token)
                .map_err(|e|self.error(FragmentCause::Host(CaptureRoutedHostError::from(e).into()),None))?;
            if rows>maximum || ownership.source_peer.is_none()&&rows!=token_window.source_tokens(){
                self.rows[index].state=State::Failed;return Err(self.error(FragmentCause::Source,None));
            }
        }
        let row=&mut self.rows[index];let target=row.routed.as_mut().expect("validated sparse slot");
        let result=match rows{Some(rows)=>target.begin_invocation(rows,token_window),None=>target.finish_invocation()};
        if let Err(cause)=result{row.state=State::Failed;return Err(self.error(FragmentCause::Host(cause.into()),None));}
        row.routed_pending=rows.is_some();if rows.is_none(){row.routed_invocations+=1;}Ok(())
    }
    /// Lend one short writer over the same source/rank/fragment owner. It checks
    /// source peer, owned expert, selected units and native range bounds; dropping
    /// it unfinished poisons the target rather than returning its destination.
    pub fn take_local_routed_writer<'t,'r>(&'t mut self,receipt:&'r PartitionCaptureReceiptPlan,fragment:usize)
        ->Result<CapturePartitionRoutedWriter<'t,'r>,PartitionFragmentDestinationError> {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken&&r.routed.is_some())
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let source=self.source.clone();let custody=self.custody.share_scheduled();
        self.rows[index].routed.as_mut().expect("validated sparse slot").writer(receipt)
            .map_err(|cause|PartitionFragmentDestinationError{cause:FragmentCause::Host(cause.into()),
                _value:None,_source:source,_custody:Some(custody)})
    }
    /// Seal actual native coverage and store the immutable sparse result under
    /// the original allowance charge. Native completion remains the caller's
    /// independent obligation; no delivery vote or global record is implied.
    pub fn finish_local_routed(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize)
        ->Result<(),PartitionFragmentDestinationError> {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        let (dtype,estimate)=self.allowance.fragment_source(producer,fragment)
            .map(|(dtype,estimate)|(dtype.clone(),estimate)).ok_or_else(||self.error(FragmentCause::Source,None))?;
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken&&r.routed.is_some())
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        if self.prefill.is_some_and(|inference|self.rows[index].routed_pending
            ||self.rows[index].routed_invocations!=inference.input_positions.div_ceil(inference.prefill_chunk_positions)) {
            self.rows[index].state=State::Failed;return Err(self.error(FragmentCause::Source,None));
        }
        let target=self.rows[index].routed.take().expect("validated sparse slot");
        let value=match target.finish(receipt) {
            Ok(value)=>value,
            Err(cause)=>{self.rows[index].state=State::Failed;return Err(self.error(cause.into(),None));}
        };
        self.record_routed(receipt,producer,fragment,&dtype,estimate.capture,value)
    }
    pub(super) fn record_routed(&mut self,receipt:&PartitionCaptureReceiptPlan,producer:usize,fragment:usize,
        dtype:&TensorDtype,charged:CaptureUsage,value:ClaimedPartitionRoutedUnits)->Result<(),PartitionFragmentDestinationError> {
        if !self.matches(receipt){return Err(self.error(FragmentCause::RoutedRecord(CaptureRunHostError::ReceiptMismatch,value),None));}
        if let Err(cause)=self.custody.validate(){return Err(self.error(FragmentCause::RoutedRecord(cause.into(),value),None));}
        let expected=self.allowance.fragment_source(producer,fragment);
        let index=self.rows.iter().position(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Taken);
        let Some(index)=index else{return Err(self.error(FragmentCause::RoutedRecord(CaptureRunHostError::ReceiptMismatch,value),None));};
        let row=&mut self.rows[index];row.state=State::Failed;
        let identity=value.identity();
        if !expected.is_some_and(|(source,estimate)|source==dtype&&estimate.capture==charged)
            ||!row.custody.same_schedule(&identity.custody)||identity.phase!=receipt.context().phase
            ||identity.prediction!=receipt.context().prediction||identity.index!=receipt.context().selection_index
            ||row.value.is_some()||row.prefill.is_some()||row.routed.is_some()||row.routed_value.is_some() {
            return Err(self.error(FragmentCause::RoutedRecord(CaptureRunHostError::ReceiptMismatch,value),None));
        }
        row.routed_value=Some(value);row.state=State::Complete;Ok(())
    }
    /// Completed sparse fragment, retaining original coordinates, source ranges
    /// and Host identity. This borrow does not imply all-rank delivery agreement.
    pub fn routed_value(&self,producer:usize,fragment:usize)->Option<&ClaimedPartitionRoutedUnits> {
        self.rows.iter().find(|r|r.producer==producer&&r.fragment==fragment&&r.state==State::Complete)?.routed_value.as_ref()
    }
}
pub(super) fn prefill_control_bytes()->Option<usize> {
    let parts=[size_of::<CaptureRoutedPrefillPlan<'_>>(),size_of::<CaptureRoutedPrefillFragment<'_,'_>>(),
        size_of::<Result<CaptureRoutedPrefillFragment<'_,'_>,CapturePrefillGeometryError>>(),size_of::<CaptureRoutedTokenWindow>()*2,size_of::<Result<CaptureRoutedPrefillPlan<'_>,CapturePrefillGeometryError>>(),
        size_of::<CapturePartitionRoutedHostPlan<'_>>(),size_of::<Result<CapturePartitionRoutedHostPlan<'_>,CaptureRunHostError>>(),
        size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry)>(),
        size_of::<Result<(),CaptureRunHostError>>(),size_of::<InferenceGeometry>(),size_of::<bool>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
pub(super) fn control_bytes()->Option<usize> {
    let parts=[prefill_control_bytes()?,CapturePartitionRoutedHostPlan::control_bytes()?,
        size_of::<PreparedPartitionRoutedCapture>(),size_of::<ClaimedPartitionRoutedUnits>(),
        size_of::<CapturePartitionRoutedClaim<'_,'_>>(),size_of::<CapturePartitionRoutedWriter<'_,'_>>(),
        size_of::<PreparedPartitionFragmentLoan>(),size_of::<Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError>>(),
        size_of::<Result<CapturePartitionRoutedWriter<'_,'_>,PartitionFragmentDestinationError>>(),
        size_of::<Result<ClaimedPartitionRoutedUnits,PartitionRoutedCaptureFailure>>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,&TensorDtype,PartitionCaptureNativeEstimate,u64)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,usize,&TensorDtype,CaptureUsage,ClaimedPartitionRoutedUnits)>(),
        size_of::<CaptureTensorCustody>()*2,size_of::<SharedCapturePlan>(),size_of::<ReceiptIdentity>(),
        size_of::<Option<usize>>(),size_of::<(TensorDtype,PartitionCaptureNativeEstimate)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,u64,Option<u64>)>(),
        size_of::<Option<u64>>(),size_of::<InferenceGeometry>(),
        size_of::<Result<(),PartitionFragmentDestinationError>>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
