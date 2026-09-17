//! Actual temporal source of the same finite projected receipt program.
use super::*;

#[derive(Clone,Copy,Debug)]
pub(super) enum Coordinate {
    Prefill(InferenceGeometry),
    Decode(u64),
    Invocation(CapturePhase,u64),
}
impl Coordinate {
    pub(super) fn control_bytes()->Option<usize>{
        let parts=[size_of::<Self>()*2,size_of::<(Self,&CaptureSelection)>(),size_of::<(Self,&PartitionCaptureContext)>(),
            size_of::<Option<InferenceGeometry>>(),size_of::<(Self,&PartitionCaptureReceiptPlan)>(),
            size_of::<Result<PartitionFragmentHostPlan<'_>,crate::working_memory::CaptureRunHostError>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(super) fn accepts(self,selection:&CaptureSelection)->bool {match self {
        Self::Prefill(g)=>g.input_positions>0&&g.prefill_chunk_positions>0
            &&selection.schedule.includes(CapturePhase::Prefill,0),
        Self::Decode(prediction)=>prediction>0&&selection.schedule.includes(CapturePhase::Decode,prediction),
        Self::Invocation(phase,prediction)=>selection.schedule.includes(phase,prediction),
    }}
    pub(super) fn matches(self,context:&PartitionCaptureContext)->bool {match self {
        Self::Prefill(_)=>context.phase==CapturePhase::Prefill&&context.prediction==0,
        Self::Decode(prediction)=>context.phase==CapturePhase::Decode&&context.prediction==prediction,
        Self::Invocation(phase,prediction)=>context.phase==phase&&context.prediction==prediction,
    }}
    pub(super) fn prefill(self)->Option<InferenceGeometry> {match self {
        Self::Prefill(g)=>Some(g),Self::Decode(_)|Self::Invocation(..)=>None,
    }}
    pub(super) fn attempts(self)->u64 {match self {
        Self::Prefill(g)=>g.input_positions.div_ceil(g.prefill_chunk_positions),Self::Decode(_)|Self::Invocation(..)=>1,
    }}
    pub(super) fn host<'a>(self,receipt:&'a PartitionCaptureReceiptPlan)
        ->Result<PartitionFragmentHostPlan<'a>,crate::working_memory::CaptureRunHostError> {
        match self {Self::Prefill(g)=>PartitionFragmentHostPlan::prepare_prefill(receipt,g),
            Self::Decode(_)|Self::Invocation(..)=>PartitionFragmentHostPlan::prepare(receipt)}
    }
}
