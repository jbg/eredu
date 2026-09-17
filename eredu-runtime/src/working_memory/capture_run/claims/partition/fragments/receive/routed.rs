//! Closed sparse destination adapter for the shared bounded receipt reader.
use super::*;

#[derive(Debug,thiserror::Error)]
pub(in crate::working_memory::capture_run::claims::partition) enum RoutedReceiveCause {
    #[error("sparse receipt source, fragment or payload differs")] Source,
    #[error(transparent)] Host(#[from] CaptureRunHostError),
    #[error(transparent)] Routed(#[from] CaptureRoutedHostError),
    #[error(transparent)] Finish(#[from] PartitionRoutedCaptureFailure),
    #[error(transparent)] Destination(#[from] PartitionFragmentDestinationError),
}
#[derive(Default)]
struct Row {peer:Option<u64>,token:u64,slot:u64,expert:u64,coefficient:f32,start:u64,stride:u64}
pub(in crate::working_memory::capture_run::claims::partition) struct RoutedReceiver<'a> {
    bank:&'a mut PreparedPartitionFragmentDestinations,receipt:&'a PartitionCaptureReceiptPlan,producer:usize,
    dtype:Option<TensorDtype>,active:Option<usize>,completed:usize,error:Option<RoutedReceiveCause>,
    geometry:RoutedUnitGeometry,units:[usize;1],start:u64,stride:u64,row:Row,range:[u64;2],written:usize,
}
impl<'a> RoutedReceiver<'a> {
    pub(in crate::working_memory::capture_run::claims::partition) fn new(bank:&'a mut PreparedPartitionFragmentDestinations,
        receipt:&'a PartitionCaptureReceiptPlan,producer:usize,dtype:Option<&TensorDtype>)->Result<Self,RoutedReceiveCause> {
        if !bank.matches(receipt)||producer==bank.allowance.local_rank()||receipt.routed_producer(producer).is_none()
            || !matches!(dtype,None|Some(TensorDtype::F16|TensorDtype::F32|TensorDtype::Bf16|TensorDtype::F64)) {
            return Err(RoutedReceiveCause::Source);
        }
        let projection=receipt.producer(producer).ok_or(RoutedReceiveCause::Source)?;
        for fragment in 0..projection.fragments().len(){
            if !bank.allowance.fragment_source(producer,fragment).is_some_and(|(precision,_)|dtype.is_none()||Some(precision)==dtype){
                return Err(RoutedReceiveCause::Source);
            }
        }
        Ok(Self{bank,receipt,producer,dtype:dtype.cloned(),active:None,completed:0,error:None,
            geometry:RoutedUnitGeometry{experts:0,units_per_expert:0,routes_per_token:0},units:[0],start:0,stride:0,
            row:Row::default(),range:[0;2],written:0})
    }
    fn remember<T>(&mut self,result:Result<T,RoutedReceiveCause>)->Option<T> {
        match result{Ok(value)=>Some(value),Err(cause)=>{if self.error.is_none(){self.error=Some(cause);}None}}
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn take_error(&mut self)->Option<RoutedReceiveCause>{self.error.take()}
    pub(in crate::working_memory::capture_run::claims::partition) fn fragments(&self)->usize {
        self.receipt.producer(self.producer).expect("retained producer").fragments().len()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn fragment_index(&self)->usize {self.completed}
    pub(in crate::working_memory::capture_run::claims::partition) fn complete(&self)->bool {
        self.error.is_none()&&self.active.is_none()&&self.completed==self.fragments()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn begin_fragment(&mut self,index:usize)
        ->Option<(CaptureUsage,[usize;32],[usize;32])> {
        let result=self.begin_fragment_worker(index);self.remember(result)
    }
    fn begin_fragment_worker(&mut self,index:usize)->Result<(CaptureUsage,[usize;32],[usize;32]),RoutedReceiveCause> {
        if self.active.is_some()||index!=self.completed||index>=self.fragments(){return Err(RoutedReceiveCause::Source);}
        let plan=CapturePartitionRoutedHostPlan::prepare(self.receipt,self.producer,index)?;
        self.geometry=plan.geometry().bank();self.units=[plan.geometry().shape()[2]];
        self.start=plan.geometry().starts()[2];self.stride=plan.geometry().strides()[2];
        let (dtype,charged,_)=self.bank.allowance.fragment_record_charge(self.receipt,self.producer,index)
            .ok_or(RoutedReceiveCause::Source)?;
        if self.dtype.as_ref().is_some_and(|actual|actual!=&dtype){return Err(RoutedReceiveCause::Source);}
        let row=self.bank.rows.iter_mut().find(|row|row.producer==self.producer&&row.fragment==index&&row.state==State::Available)
            .ok_or(RoutedReceiveCause::Source)?;
        row.state=State::Taken;
        let identity=ReceiptIdentity{phase:self.receipt.context().phase,prediction:self.receipt.context().prediction,
            index:self.receipt.context().selection_index,custody:row.custody.share_scheduled()};
        match CapturePartitionRoutedClaim::from_plan(plan,identity).prepare_received(){
            Ok(target)=>row.routed=Some(target),Err(cause)=>{row.state=State::Failed;return Err(cause.into());}
        }
        self.active=Some(index);
        let projection=self.receipt.producer(self.producer).expect("retained producer");
        let mut source=[0;32];let mut selected=[0;32];
        for axis in 0..3 {
            source[axis]=usize::try_from(projection.local_shape()[axis]).map_err(|_|RoutedReceiveCause::Source)?;
            selected[axis]=usize::try_from(projection.fragments()[index].local().shape[axis]).map_err(|_|RoutedReceiveCause::Source)?;
        }
        Ok((charged,source,selected))
    }
    fn target(&mut self)->Result<&mut PreparedPartitionRoutedCapture,RoutedReceiveCause> {
        let fragment=self.active.ok_or(RoutedReceiveCause::Source)?;
        self.bank.rows.iter_mut().find(|row|row.producer==self.producer&&row.fragment==fragment&&row.state==State::Taken)
            .and_then(|row|row.routed.as_mut()).ok_or(RoutedReceiveCause::Source)
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn geometry_scalar(&self,field:usize,value:u64)->bool {
        match field{0=>self.geometry.experts==value,1=>self.geometry.units_per_expert==value,2=>self.geometry.routes_per_token==value,_=>false}
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn begin_range(&mut self){self.range=[0;2];}
    pub(in crate::working_memory::capture_run::claims::partition) fn range_scalar(&mut self,index:usize,value:u64)->bool {
        if let Some(field)=self.range.get_mut(index){*field=value;true}else{false}
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn finish_range(&mut self)->bool {
        let [start,end]=self.range;let result=self.target().and_then(|target|target.received_chunk(start,end).map_err(Into::into));
        self.remember(result).is_some()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn begin_row(&mut self){self.row=Row::default();self.written=0;}
    pub(in crate::working_memory::capture_run::claims::partition) fn row_peer(&mut self,value:Option<u64>){self.row.peer=value;}
    pub(in crate::working_memory::capture_run::claims::partition) fn row_integer(&mut self,field:usize,value:u64)->bool {
        match field {1=>self.row.token=value,2=>self.row.slot=value,3=>self.row.expert=value,
            5=>self.row.start=value,6=>self.row.stride=value,_=>return false}true
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn row_coefficient(&mut self,value:f32){self.row.coefficient=value;}
    pub(in crate::working_memory::capture_run::claims::partition) fn begin_values(&mut self)->bool {
        let result=(||{
            if self.row.start!=self.start||self.row.stride!=self.stride{return Err(RoutedReceiveCause::Source);}
            let row=&self.row;let (peer,token,slot,expert,coefficient)=(row.peer,row.token,row.slot,row.expert,row.coefficient);
            let receipt=self.receipt;self.target()?.received_begin_row(receipt,peer,token,slot,expert,coefficient)?;Ok(())
        })();self.remember(result).is_some()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn shape(&self)->&[usize]{&self.units}
    pub(in crate::working_memory::capture_run::claims::partition) fn len(&self)->usize{self.units[0]}
    pub(in crate::working_memory::capture_run::claims::partition) fn push(&mut self,value:f32)->bool {
        let result=self.target().and_then(|target|target.received_push(value).map_err(Into::into));
        if self.remember(result).is_some(){self.written+=1;true}else{false}
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn finish_row(&mut self)->bool {
        let result=(||{if self.written!=self.units[0]{return Err(RoutedReceiveCause::Source);}
            self.target()?.received_finish_row()?;Ok(())})();self.remember(result).is_some()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn finish_fragment(&mut self)->bool {
        let result=(||{
            let fragment=self.active.ok_or(RoutedReceiveCause::Source)?;
            let row=self.bank.rows.iter_mut().find(|row|row.producer==self.producer&&row.fragment==fragment&&row.state==State::Taken)
                .ok_or(RoutedReceiveCause::Source)?;
            let target=row.routed.take().ok_or(RoutedReceiveCause::Source)?;
            let value=match target.finish(self.receipt){Ok(value)=>value,Err(cause)=>{row.state=State::Failed;return Err(cause.into());}};
            if self.dtype.is_none()&&(value.native_rows()!=0||!value.observation().rows.is_empty()||!value.observation().source_token_ranges.is_empty()) {
                row.state=State::Failed;return Err(RoutedReceiveCause::Source);
            }
            let (dtype,estimate)=self.bank.allowance.fragment_source(self.producer,fragment)
                .map(|(dtype,estimate)|(dtype.clone(),estimate)).ok_or(RoutedReceiveCause::Source)?;
            self.bank.record_routed(self.receipt,self.producer,fragment,&dtype,estimate.capture,value)?;
            self.active=None;self.completed+=1;Ok(())
        })();self.remember(result).is_some()
    }
    pub(in crate::working_memory::capture_run::claims::partition) fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<Row>()*2,size_of::<Result<Self,RoutedReceiveCause>>(),
            size_of::<RoutedReceiveCause>()*2,size_of::<Option<RoutedReceiveCause>>(),
            size_of::<Result<(CaptureUsage,[usize;32],[usize;32]),RoutedReceiveCause>>(),
            size_of::<[[usize;32];2]>(),size_of::<CaptureUsage>(),size_of::<CapturePartitionRoutedHostPlan<'_>>(),
            size_of::<CapturePartitionRoutedClaim<'_,'_>>(),size_of::<ReceiptIdentity>(),
            size_of::<PreparedPartitionRoutedCapture>(),size_of::<Result<PreparedPartitionRoutedCapture,CaptureRunHostError>>(),
            size_of::<Result<ClaimedPartitionRoutedUnits,PartitionRoutedCaptureFailure>>(),
            size_of::<Result<(),RoutedReceiveCause>>(),size_of::<(TensorDtype,PartitionCaptureNativeEstimate)>(),
            size_of::<(Option<u64>,u64,u64,u64,f32)>(),size_of::<[u64;2]>(),
            size_of::<(&mut Self,usize)>(),size_of::<(&mut Self,u64)>(),size_of::<(&mut Self,f32)>(),
            size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,usize,Option<&TensorDtype>)>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
