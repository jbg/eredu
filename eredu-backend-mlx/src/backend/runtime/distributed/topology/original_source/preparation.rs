//! Exact U32 world frames use the same initialized leaf/accepted Gather workers.
use super::*;
use safemlx::{PreparedInputRuntime,PreparedInputPlan,OriginalScopeObserver,distributed::GroupWorkerOperation};
use crate::backend::runtime::distributed::completion::{prepared::ReadyCompletionResources,
    PreparedCommunicationU32Words,OriginalCommunicationU32Words,MlxNeuralCommunicationCompletion};

// The final Arc allocation retires before its Q; every clone remains payload-free.
#[derive(Debug)]
pub(super) struct SharedPreparationCustody(Option<std::sync::Arc<eredu_runtime::working_memory::CommunicationPreparationCustody>>);
impl SharedPreparationCustody {
    pub(super) fn new(value:eredu_runtime::working_memory::CommunicationPreparationCustody)->Self {
        Self(Some(std::sync::Arc::new(value)))
    }
    pub(super) fn allocation_bytes()->Option<usize> {
        std::alloc::Layout::new::<[usize;2]>()
            .extend(std::alloc::Layout::new::<eredu_runtime::working_memory::CommunicationPreparationCustody>())
            .ok().map(|(layout,_)|layout.pad_to_align().size())
    }
}
impl Clone for SharedPreparationCustody {
    fn clone(&self)->Self{Self(self.0.clone())}
}
impl Drop for SharedPreparationCustody {
    fn drop(&mut self){if let Some(value)=self.0.take(){drop(std::sync::Arc::into_inner(value));}}
}

/// One actual shared-coordinator frame. Its immutable source backing and metadata
/// are paid before any native operation is constructed; no model role is implied.
pub(crate) struct OriginalPreparationFrame {
    value:eredu_runtime::working_memory::PreparedCommunicationPreparation<Array>,
    word_count:usize,
    runtime:PreparedInputRuntime,
    source:RetainedCommunicationSource,
    funding:HostMetadataFunding,
}
/// One actual world gather, including its preallocated unsigned host destination.
pub(crate) struct PreparedOriginalPreparationGather<'a> {
    operation:OriginalCommunicationCompletedOperation<'a>,
    words:PreparedCommunicationU32Words,
    ready:ReadyCompletionResources,
    runtime:&'a PreparedInputRuntime,
    backing:usize,
    source:RetainedCommunicationSource,
    funding:HostMetadataFunding,
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn prepare_preparation_frame(&self,
        words:&[u32],
        pool:&eredu_runtime::working_memory::WorkingMemoryPool,
        policy:Option<&eredu_runtime::working_memory::SessionResetPreparationFunding>)->Result<OriginalPreparationFrame,Error> {
        reserve(self.funding(),&[
            size_of::<OriginalPreparationFrame>(),size_of::<Result<OriginalPreparationFrame,Error>>(),
            size_of::<(&Self,&[u32],&eredu_runtime::working_memory::WorkingMemoryPool,Option<&eredu_runtime::working_memory::SessionResetPreparationFunding>)>(),
            size_of::<PreparedInputRuntime>(),size_of::<Result<PreparedInputRuntime,eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<PreparedInputPlan<'_>>(),size_of::<Result<PreparedInputPlan<'_>,safemlx::PreparedInputCause>>(),
            size_of::<[usize;1]>(),size_of::<[Array;1]>(),
            safemlx::InitializedInputAllocator::borrow_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<RetiredFailure>().ok_or_else(overflow)?,
            size_of::<FrameProducer<'_, '_>>(),size_of::<RetiredFailure>(),
            size_of::<Result<eredu_runtime::working_memory::PreparedCommunicationPreparation<Array>,
                eredu_runtime::working_memory::CommunicationPreparationError<FrameProducer<'_, '_>>>>(),
        ])?;
        self.validate()?;
        let runtime=crate::backend::managed_memory::input_allocator::borrow_admitted(pool)
            .map_err(|cause|failure(Cause::Allocator(cause),self.source(),self.funding()))?;
        let shape=[words.len()];
        let plan=runtime.u32(words,&shape).map_err(|cause|failure(Cause::Input(cause),self.source(),self.funding()))?;
        let value=pool.prepare_communication(FrameProducer{source:self,plan,words,pool,policy})
            .map_err(|cause|retired_error(cause.retire(),self.source(),self.funding()))?;
        Ok(OriginalPreparationFrame{value,word_count:words.len(),runtime,source:self.source().clone(),funding:self.funding().clone()})
    }
}
impl OriginalPreparationFrame {
    pub(crate) fn prepare<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>)
        ->Result<PreparedOriginalPreparationGather<'a>,Error> {
        reserve(&self.funding,&[
            size_of::<PreparedOriginalPreparationGather<'a>>(),size_of::<Result<PreparedOriginalPreparationGather<'a>,Error>>(),
            size_of::<(&Self,&OriginalCommunicationSource<'_>)>(),size_of::<(usize,usize)>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        if !self.source.same_source(source.source()) || !self.funding.same_account(source.funding()) {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        let operation=source.world_cpu_operation_storage(self.value.output(),GroupWorkerOperation::Gather)?;
        let expected=self.word_count.checked_mul(source.source().manifest().world_size())
            .ok_or_else(overflow)?;
        if operation.native().constructor().output_geometry()!=(1,expected)
            || operation.native().constructor().output_dtype()!=safemlx::Dtype::Uint32 {
            return Err(failure(Cause::Output,&self.source,&self.funding));
        }
        let backing=operation.backing_storage(&self.runtime)?.capacity();
        let operation=operation.with_completion()?;
        let words=PreparedCommunicationU32Words::prepare_operation(source,&operation)?;
        let ready=operation.prepare_resources(source,None)?;
        Ok(PreparedOriginalPreparationGather{operation,words,ready,runtime:&self.runtime,backing,
            source:self.source.clone(),funding:self.funding.clone()})
    }
}
impl PreparedOriginalPreparationGather<'_> {
    pub(crate) fn graph_capacity(&self)->usize{self.operation.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.operation.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize{self.backing}
    pub(crate) fn runtime(&self)->&PreparedInputRuntime{self.runtime}
    pub(crate) fn submit(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<(OriginalCommunicationU32Words,MlxNeuralCommunicationCompletion),Error> {
        reserve(&self.funding,&[size_of::<Self>(),size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            size_of::<Result<(OriginalCommunicationU32Words,MlxNeuralCommunicationCompletion),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let accepted=self.operation.construct_accepted(source,observer,stream)?;
        let (words,completion)=self.words.submit_accepted(accepted,self.ready)?;
        Ok((words,completion.into()))
    }
}
fn overflow()->Error{Error::WorkspacePlanning(HostMetadataFundingError::Overflow)}
fn reserve(funding:&HostMetadataFunding,bytes:&[usize])->Result<(),Error>{
    funding.reserve_metadata(bytes.iter().copied().try_fold(size_of_val(bytes),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}

struct FrameProducer<'a,'native>{
    source:&'a OriginalCommunicationSource<'native>,plan:PreparedInputPlan<'a>,
    words:&'a [u32],
    pool:&'a eredu_runtime::working_memory::WorkingMemoryPool,
    policy:Option<&'a eredu_runtime::working_memory::SessionResetPreparationFunding>,
}
impl eredu_runtime::working_memory::CommunicationPreparationProducer for FrameProducer<'_, '_>{
    type Output=Array;type Error=Error;
    fn source(&self)->&RetainedCommunicationSource{self.source.source()}
    fn frame(&self)->&[u32]{self.words}
    fn validate_pool(&self,pool:&eredu_runtime::working_memory::WorkingMemoryPool)->Result<(),eredu_runtime::working_memory::WorkingMemoryError>{
        if self.pool.same_domain(pool){Ok(())}else{Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)}
    }
    fn required_storage_bytes(&self)->Result<usize,eredu_runtime::working_memory::WorkingMemoryError>{
        super::inputs::storage_bytes(std::array::from_ref(&self.plan))
    }
    fn check_preparation_policy(&self,pool:&eredu_runtime::working_memory::WorkingMemoryPool,bytes:u64)
        ->Result<(),eredu_runtime::working_memory::WorkingMemoryError>{
        self.policy.map_or(Ok(()),|policy|policy.charge_communication(pool,bytes))
    }
    fn produce(self,custody:eredu_runtime::working_memory::CommunicationPreparationCustody)->Result<Array,Error>{
        let [value]=super::inputs::construct_admitted(self.source,[self.plan],custody)?;Ok(value)
    }
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct RetiredFailure {
    #[source] cause:eredu_runtime::working_memory::RetiredCommunicationPreparationError<Error>,
    source:RetainedCommunicationSource,funding:HostMetadataFunding,
}
fn retired_error(cause:eredu_runtime::working_memory::RetiredCommunicationPreparationError<Error>,
    source:&RetainedCommunicationSource,funding:&HostMetadataFunding)->Error{
    Error::with_original_control_source(eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Other,
        RetiredFailure{cause,source:source.clone(),funding:funding.clone()}),false)
}
mod submit;
pub(crate) use submit::PreparationCompletion;
