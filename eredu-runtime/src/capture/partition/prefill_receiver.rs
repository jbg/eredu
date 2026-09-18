//! Real local lazy source for a producer whose selected spatial overlap is empty.
use super::*;
use eredu_core::{InferenceGeometry,checkpoint::TensorDtype};
use std::mem::{size_of,size_of_val};

/// Owned fixed metadata used while the backend borrows its existing completion
/// worker. This has no destination, synthetic fragment, quota or native grant.
#[derive(Debug)]
pub struct PartitionPrefillReceiverSource {
    shape:[usize;32],rank:usize,dtype:TensorDtype,
    selection:usize,producer:usize,chunk:u64,forward_epoch:u64,
    source:SharedCapturePlan,
}
impl PartitionPrefillReceiverSource {
    pub(crate) fn prepare(receipt:&PartitionCaptureReceiptPlan,producer:usize,
        dtype:TensorDtype,inference:InferenceGeometry,chunk:u64)
        ->Result<Self,PartitionPrefillCaptureSourceError> {
        use PartitionPrefillCaptureSourceError as E;
        let projection=receipt.producer(producer).ok_or(E::Source)?;
        if !projection.fragments().is_empty(){return Err(E::Source);}
        Self::prepare_local(receipt,producer,dtype,inference,chunk,projection.local_shape())
    }
    pub(crate) fn prepare_local(receipt:&PartitionCaptureReceiptPlan,producer:usize,
        dtype:TensorDtype,inference:InferenceGeometry,chunk:u64,local_shape:&[u64])
        ->Result<Self,PartitionPrefillCaptureSourceError> {
        use PartitionPrefillCaptureSourceError as E;
        let context=receipt.context();let source=receipt.shared_plan_source();
        if context.phase!=CapturePhase::Prefill || context.prediction!=0 || context.invocation.is_some()
            || context.capture_plan_identity!=source.admission().identity() || receipt.routed_producer(producer).is_some()
            || producer>=receipt.world_size() || local_shape.is_empty() || local_shape.len()>32
            || !matches!(dtype,TensorDtype::F32|TensorDtype::F16|TensorDtype::Bf16) {return Err(E::Source);}
        let selection=source.admission().plan().selections.get(context.selection_index).ok_or(E::Source)?;
        // Reuse the same ordinary temporal declaration. No local capture slice
        // is manufactured merely to inspect this physical source.
        let axis=match selection.transform {
            CaptureTransform::Summary|CaptureTransform::Histogram{..}=>
                CapturePrefillTransformPlan::prepare(source.admission(),context.selection_index,inference)
                    .map_err(CapturePrefillPartitionError::from)?.window().axis(),
            CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..}=>
                CapturePrefillRowAssembly::prepare(source.admission(),context.selection_index,inference)
                    .map_err(CapturePrefillPartitionError::from)?.sequence_axis(),
            _=>return Err(E::Source),
        };
        if axis>=local_shape.len() || local_shape[axis]!=inference.input_positions
            || chunk>=inference.input_positions.div_ceil(inference.prefill_chunk_positions) {return Err(E::Source);}
        let start=chunk.checked_mul(inference.prefill_chunk_positions).ok_or(E::Source)?;
        let length=inference.prefill_chunk_positions.min(inference.input_positions-start);
        let mut shape=[0usize;32];let rank=local_shape.len();
        for (axis,&value) in local_shape.iter().enumerate(){shape[axis]=usize::try_from(value).map_err(|_|E::Source)?;}
        shape[axis]=usize::try_from(length).map_err(|_|E::Source)?;
        Ok(Self{shape,rank,dtype,selection:context.selection_index,producer,chunk,
            forward_epoch:context.forward_epoch,source:source.clone()})
    }
    /// Exact local physical chunk shape, independent of zero selected overlap.
    pub fn source_shape(&self)->&[usize]{&self.shape[..self.rank]}
    /// Scalar supplied by the retained and coordinated actual producer source.
    pub fn dtype(&self)->&TensorDtype{&self.dtype}
    /// Original request, selection, rank and canonical chunk identity.
    pub(crate) fn matches(&self,source:&SharedCapturePlan,selection:usize,producer:usize,chunk:u64,epoch:u64)->bool {
        self.source.same_storage(source) && self.selection==selection && self.producer==producer
            && self.chunk==chunk && self.forward_epoch==epoch
    }
    pub(crate) fn control_bytes()->Option<usize> {
        let parts=[size_of::<Self>()*2,size_of::<Result<Self,PartitionPrefillCaptureSourceError>>(),
            size_of::<PartitionPrefillCaptureSourceError>(),size_of::<[usize;32]>(),size_of::<usize>()*3,
            size_of::<u64>()*3,size_of::<TensorDtype>(),size_of::<SharedCapturePlan>(),
            size_of::<(&PartitionCaptureReceiptPlan,usize,TensorDtype,InferenceGeometry,u64)>(),
            size_of::<(&PartitionCaptureReceiptPlan,usize,TensorDtype,InferenceGeometry,u64,&[u64])>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_,u64>>>(),size_of::<(usize,&u64)>(),
            CapturePrefillRowAssembly::partition_preparation_control_bytes()?,
            CapturePrefillTransformPlan::partition_preparation_control_bytes()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
