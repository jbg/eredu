//! Typed local partial statistics/bins through the existing reduction workers.
use super::*;

pub(super) fn selected(receipt:&PartitionCaptureReceiptPlan)->bool {
    receipt.combination()==PartitionCaptureCombination::Disjoint && receipt.shared_plan_source().is_some_and(|source|
        matches!(source.admission().plan().selections[receipt.context().selection_index].transform,
            CaptureTransform::Summary|CaptureTransform::Histogram{..}))
}
pub(super) fn source<'a>(receipt:&'a PartitionCaptureReceiptPlan,producer:usize,fragment:usize,inference:InferenceGeometry)
    ->Result<CapturePrefillTransformPlan<'a>,CaptureRunHostError> {
    if !selected(receipt)||receipt.context().phase!=CapturePhase::Prefill||receipt.context().prediction!=0
        ||receipt.context().invocation.is_some(){return Err(CaptureRunHostError::ReceiptMismatch);}
    CapturePrefillTransformPlan::prepare_partition(receipt.shared_plan_source().expect("checked source").admission(),
        receipt.context().selection_index,inference,receipt.producer(producer).ok_or(CaptureRunHostError::ReceiptMismatch)?,
        fragment,PartitionCaptureCombination::Disjoint).map_err(CapturePrefillHostError::from).map_err(Into::into)
}
fn validate_chunk(receipt:&PartitionCaptureReceiptPlan,producer:usize,fragment:usize,inference:InferenceGeometry,
    chunk:&CapturePrefillTransformFragment<'_,'_>)->Result<(),CaptureRunHostError> {
    let plan=source(receipt,producer,fragment,inference)?;
    if !std::ptr::eq(plan.admission(),chunk.plan().admission())||plan.selection_index()!=chunk.plan().selection_index()
        ||inference!=chunk.plan().inference_geometry(){return Err(CaptureRunHostError::ReceiptMismatch);}
    let expected=plan.fragment(chunk.chunk_index()).map_err(CaptureStepError::from)?;
    if expected.input()!=chunk.input()||expected.position()!=chunk.position()||expected.output_demand()!=chunk.output_demand()
        ||expected.source_shape()!=chunk.source_shape()||expected.selected_shape()!=chunk.selected_shape()
        ||expected.starts()!=chunk.starts()||expected.ends()!=chunk.ends()||expected.strides()!=chunk.strides()
        ||expected.selected_elements()!=chunk.selected_elements(){return Err(CaptureRunHostError::ReceiptMismatch);}
    Ok(())
}
impl PreparedPartitionFragmentDestinations {
    /// Issue the existing typed native Summary/Histogram claim for exactly one
    /// reached local chunk. It retains the same stamped-source transfer methods.
    /// Dropped or mismatched receipts leave the pending attempt non-reusable.
    pub fn take_local_prefill_reduction<'a,'t>(&'t mut self,receipt:&'a PartitionCaptureReceiptPlan,fragment:usize,
        chunk:&CapturePrefillTransformFragment<'_,'_>)
        ->Result<PartitionFragmentDestination<'a,'t>,PartitionFragmentDestinationError> {
        self.custody.validate().map_err(|e|self.error(e.into(),None))?;
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,None));}
        let producer=self.allowance.local_rank();
        validate_chunk(receipt,producer,fragment,inference,chunk).map_err(|e|self.error(e.into(),None))?;
        let index=self.rows.iter().position(|row|row.producer==producer&&row.fragment==fragment&&row.state==State::Taken
            &&row.prefill.as_ref().is_some_and(|target|target.next==chunk.chunk_index()&&!target.pending&&!target.slot.done
                &&matches!(target.slot.state,TargetState::Claimed|TargetState::Active)))
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        self.rows[index].prefill.as_mut().expect("checked target").pending=true;
        let identity=ReceiptIdentity{phase:CapturePhase::Prefill,prediction:0,index:receipt.context().selection_index,
            custody:self.rows[index].custody.share_scheduled()};
        let source=receipt.shared_plan_source().expect("checked source").admission();
        let projection=receipt.producer(producer).expect("checked producer");
        match source.plan().selections[receipt.context().selection_index].transform {
            CaptureTransform::Summary=>{
                let geometry=CaptureSummaryGeometry::prepare_partition(source,receipt.context().selection_index,CapturePhase::Prefill,0,None,
                    projection,fragment).and_then(|geometry|geometry.fragment(chunk))
                    .map_err(CaptureStepError::from).map_err(|e|self.error(CaptureRunHostError::from(e).into(),None))?;
                let plan=CaptureSummaryHostPlan::prepare(geometry).map_err(|e|self.error(e.into(),None))?;
                Ok(PartitionFragmentDestination::Summary(CaptureSummaryClaim::from_partition_prefill_plan(plan,identity,chunk.chunk_index())))
            },
            CaptureTransform::Histogram{..}=>{
                let geometry=CaptureHistogramGeometry::prepare_partition(source,receipt.context().selection_index,CapturePhase::Prefill,0,None,
                    projection,fragment).and_then(|geometry|geometry.fragment(chunk))
                    .map_err(CaptureStepError::from).map_err(|e|self.error(CaptureRunHostError::from(e).into(),None))?;
                let plan=CaptureHistogramHostPlan::prepare(geometry).map_err(|e|self.error(e.into(),None))?;
                Ok(PartitionFragmentDestination::Histogram(CaptureHistogramClaim::from_partition_prefill_plan(plan,identity,chunk.chunk_index())))
            },_=>Err(self.error(FragmentCause::Source,None)),
        }
    }
    /// Validate source, chunk, counts, bins and custody before appending through
    /// the same fixed Summary/Histogram arithmetic used by ordinary prefill.
    pub fn record_local_prefill_reduction(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize,
        chunk:&CapturePrefillTransformFragment<'_,'_>,value:PartitionFragmentValue)
        ->Result<(),PartitionFragmentDestinationError> {
        if let Err(error)=self.custody.validate(){return Err(self.error(error.into(),Some(value)));}
        let Some(inference)=self.prefill else{return Err(self.error(FragmentCause::Source,Some(value)));};
        if !self.matches(receipt){return Err(self.error(FragmentCause::Source,Some(value)));}
        let producer=self.allowance.local_rank();
        if let Err(error)=validate_chunk(receipt,producer,fragment,inference,chunk){return Err(self.error(error.into(),Some(value)));}
        let Some(index)=self.rows.iter().position(|row|row.producer==producer&&row.fragment==fragment&&row.state==State::Taken)
            else{return Err(self.error(FragmentCause::Source,Some(value)));};
        self.rows[index].state=State::Failed;
        let result=(||->Result<(),CaptureRunHostError>{
            let row=&mut self.rows[index];let identity=value.identity();
            if !row.custody.same_schedule(&identity.custody)||identity.phase!=CapturePhase::Prefill||identity.prediction!=0
                ||identity.index!=receipt.context().selection_index{return Err(CaptureRunHostError::ReceiptMismatch);}
            let target=row.prefill.as_mut().ok_or(CaptureRunHostError::ReceiptMismatch)?;
            if target.next!=chunk.chunk_index()||!target.pending||target.slot.done
                ||!matches!(target.slot.state,TargetState::Claimed|TargetState::Active){return Err(CaptureRunHostError::ReceiptMismatch);}
            match (&value,&chunk.plan().selection().transform){
                (PartitionFragmentValue::Summary(value),CaptureTransform::Summary) if value.partition_chunk()==Some(target.next)=>{
                    let empty=crate::capture::reduction::Summary::default();
                    target.slot.summary=Some(target.slot.summary.as_ref().unwrap_or(&empty)
                        .appended_fixed(value.observation(),chunk.selected_elements()).map_err(CaptureStepError::from)?);
                },
                (PartitionFragmentValue::Histogram(value),CaptureTransform::Histogram{edges}) if value.partition_chunk()==Some(target.next)=>{
                    crate::capture::reduction::validate_histogram_fixed(value.observation(),edges,chunk.selected_elements())
                        .map_err(CaptureStepError::from)?;
                    if let Some(out)=&mut target.slot.histogram {
                        crate::capture::reduction::add_histogram_fixed(out,value.observation()).map_err(CaptureStepError::from)?;
                    }
                },_=>return Err(CaptureRunHostError::ReceiptMismatch),
            }Ok(())
        })();
        if let Err(error)=result{return Err(self.error(error.into(),Some(value)));}
        let row=&mut self.rows[index];let target=row.prefill.as_mut().expect("validated target");
        if let PartitionFragmentValue::Histogram(value)=value {
            if target.slot.histogram.is_none(){target.slot.histogram=Some(value.into_partition_histogram());}
        }
        target.pending=false;target.slot.done=true;target.slot.state=TargetState::Active;row.state=State::Taken;Ok(())
    }
    pub(super) fn finish_local_prefill_reduction(&mut self,receipt:&PartitionCaptureReceiptPlan,fragment:usize)
        ->Result<(),PartitionFragmentDestinationError> {
        let inference=self.prefill.ok_or_else(||self.error(FragmentCause::Source,None))?;
        let producer=self.allowance.local_rank();
        let (dtype,estimate)=self.allowance.fragment_source(producer,fragment).map(|(dtype,estimate)|(dtype.clone(),estimate))
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        let index=self.rows.iter().position(|row|row.producer==producer&&row.fragment==fragment&&row.state==State::Taken)
            .ok_or_else(||self.error(FragmentCause::Source,None))?;
        self.rows[index].state=State::Failed;
        let row=&self.rows[index];
        let valid=row.prefill.as_ref().is_some_and(|target|target.next==inference.input_positions.div_ceil(inference.prefill_chunk_positions)
            &&!target.pending&&!target.slot.done&&target.slot.state==TargetState::Active&&target.slot.dtype.as_ref()==Some(&dtype));
        if !valid{return Err(self.error(FragmentCause::Source,None));}
        let identity=ReceiptIdentity{phase:CapturePhase::Prefill,prediction:0,index:receipt.context().selection_index,custody:row.custody.share_scheduled()};
        let plan=FragmentHostPlan::prepare(receipt,producer,fragment).map_err(|e|self.error(e.into(),None))?;
        let value=match plan {
            FragmentHostPlan::Summary(plan)=>{
                let value=self.rows[index].prefill.as_ref().and_then(|target|target.slot.summary.as_ref())
                    .ok_or_else(||self.error(FragmentCause::Source,None))?.value();
                PartitionFragmentValue::Summary(CaptureSummaryClaim::from_partition_plan(plan,identity).finish_partition(value)
                    .map_err(|error|self.error(error.into(),None))?)
            },
            FragmentHostPlan::Histogram(plan)=>{
                let value=self.rows[index].prefill.as_mut().and_then(|target|target.slot.histogram.take())
                    .ok_or_else(||self.error(FragmentCause::Source,None))?;
                PartitionFragmentValue::Histogram(CaptureHistogramClaim::from_partition_plan(plan,identity).finish_partition_accumulator(value)
                    .map_err(|error|self.error(error.into(),None))?)
            },_=>return Err(self.error(FragmentCause::Source,None)),
        };
        self.rows[index].prefill=None;self.rows[index].state=State::Taken;
        self.record(receipt,producer,fragment,&dtype,estimate.capture,value)
    }
}
pub(super) fn control_bytes(receipt:&PartitionCaptureReceiptPlan)->Option<usize> {
    let histogram=matches!(receipt.shared_plan_source()?.admission().plan().selections[receipt.context().selection_index].transform,
        CaptureTransform::Histogram{..});
    let typed=if histogram {
        CaptureHistogramGeometry::preparation_control_bytes()?.checked_add(size_of::<CaptureHistogramHostPlan<'_>>() *2)?
            .checked_add(size_of::<CaptureHistogramClaim<'_,'_>>() *2)?.checked_add(size_of::<ClaimedCaptureHistogram>() *2)?
    }else{
        CaptureSummaryGeometry::preparation_control_bytes()?.checked_add(size_of::<CaptureSummaryHostPlan<'_>>() *2)?
            .checked_add(size_of::<CaptureSummaryClaim<'_,'_>>() *2)?.checked_add(size_of::<ClaimedCaptureSummary>() *2)?
    };
    let parts=[typed,CaptureTensorGeometry::partition_preparation_control_bytes()?,size_of::<Target>()*2,size_of::<TargetSlot>(),size_of::<InferenceGeometry>()*2,
        CapturePrefillTransformPlan::partition_preparation_control_bytes()?,size_of::<CapturePrefillTransformPlan<'_>>()*2,
        size_of::<CapturePrefillTransformFragment<'_,'_>>()*2,size_of::<Result<CapturePrefillTransformPlan<'_>,CaptureRunHostError>>(),
        size_of::<Result<(),CaptureRunHostError>>(),size_of::<Result<(),PartitionFragmentDestinationError>>(),
        size_of::<PartitionFragmentDestination<'_,'_>>()*2,size_of::<PartitionFragmentValue>()*2,
        size_of::<Result<PartitionFragmentDestination<'_,'_>,PartitionFragmentDestinationError>>(),
        size_of::<ReceiptIdentity>()*2,size_of::<CaptureTensorCustody>()*2,size_of::<CaptureUsage>(),size_of::<TensorDtype>(),
        size_of::<PreparedPartitionFragmentLoan>()*2,size_of::<Result<PreparedPartitionFragmentLoan,PartitionFragmentDestinationError>>(),
        size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,&TensorDtype,PartitionCaptureNativeEstimate)>(),
        size_of::<(&PartitionCaptureReceiptPlan,usize,usize,InferenceGeometry,&CapturePrefillTransformFragment<'_,'_>)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,&CapturePrefillTransformFragment<'_,'_>,PartitionFragmentValue)>(),
        size_of::<(usize,u64,bool)>(),size_of::<Option<usize>>(),size_of::<Option<&mut Target>>(),
    ];parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
