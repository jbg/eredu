//! Exact local source for one decode replica or empty-overlap producer.
use super::*;
use eredu_core::checkpoint::TensorDtype;
use std::mem::{size_of,size_of_val};

/// Fixed metadata lent while the backend borrows its existing completion
/// worker. It owns no payload, fragment destination, native quota or authority.
#[derive(Debug)]
pub struct PartitionInvocationReceiverSource {
    shape:[usize;32],rank:usize,dtype:TensorDtype,
    _source:SharedCapturePlan,
}
impl PartitionInvocationReceiverSource {
    pub(crate) fn prepare_local(receipt:&PartitionCaptureReceiptPlan,producer:usize,
        dtype:TensorDtype,shape:&[u64])->Result<Self,PartitionInvocationCaptureSourceError> {
        use PartitionInvocationCaptureSourceError as E;
        let source=receipt.shared_plan_source().ok_or(E::Source)?;
        let context=receipt.context();
        if context.phase!=CapturePhase::Decode||context.prediction==0||context.invocation.is_some()
            ||context.capture_plan_identity!=source.admission().identity()||producer>=receipt.world_size()
            ||receipt.routed_producer(producer).is_some()
            ||receipt.producer(producer).is_some_and(|p|!p.fragments().is_empty())
            ||shape.is_empty()||shape.len()>32||!matches!(dtype,TensorDtype::F32|TensorDtype::F16|TensorDtype::Bf16)
            ||!source.admission().plan().selections.get(context.selection_index)
                .is_some_and(|selection|selection.schedule.includes(context.phase,context.prediction)) {
            return Err(E::Source);
        }
        // The source constructor already authenticated this full local shape
        // against its architecture projection before the shared scalar vote.
        // A selected empty overlap is distinct from an absent physical source.
        let mut local=[0;32];
        for (index,&n) in shape.iter().enumerate(){local[index]=usize::try_from(n).map_err(|_|E::Source)?;}
        Ok(Self{shape:local,rank:shape.len(),dtype,_source:source.clone()})
    }
    /// Exact physical local dimensions before any capture selection.
    pub fn source_shape(&self)->&[usize]{&self.shape[..self.rank]}
    /// Agreed actual floating source scalar.
    pub fn dtype(&self)->&TensorDtype{&self.dtype}
    pub(crate) fn control_bytes()->Option<usize>{
        let parts=[size_of::<Self>()*2,size_of::<Result<Self,PartitionInvocationCaptureSourceError>>(),
            size_of::<PartitionInvocationCaptureSourceError>(),size_of::<(&PartitionCaptureReceiptPlan,usize,TensorDtype,&[u64])>(),
            size_of::<[usize;32]>(),size_of::<(usize,&u64)>(),size_of::<std::iter::Enumerate<std::slice::Iter<'_,u64>>>(),
            size_of::<SharedCapturePlan>(),size_of::<TensorDtype>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
