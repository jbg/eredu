//! Control-only requests retain actual initialized agreement sources. They are
//! never represented as model tensor sources or an AllReduceSum selection.
use super::*;
use super::super::OriginalCommunicationOwner;

struct Body {
    inputs:agreement::OriginalAgreementInputs,
    owner:OriginalCommunicationOwner,
    taken:Cell<bool>,
    retained:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
/// Closed initialized source for actual session agreements without a TP lane.
pub(crate) struct OriginalParallelControlSource {
    body:Option<Rc<Body>>,
    funding:WorkspaceMetadataFunding,
}
impl Clone for OriginalParallelControlSource {
    fn clone(&self)->Self{Self{body:self.body.clone(),funding:self.funding.clone()}}
}
impl Drop for OriginalParallelControlSource {
    fn drop(&mut self){if let Some(body)=self.body.take(){drop(Rc::into_inner(body));}}
}
impl std::fmt::Debug for OriginalParallelControlSource {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {
        f.debug_struct("OriginalParallelControlSource")
            .field("agreement",&self.body().inputs.group()).finish_non_exhaustive()
    }
}
impl OriginalParallelControlSource {
    fn body(&self)->&Body{self.body.as_deref().expect("live control source")}
    pub(crate) fn capacity(&self)->Result<AgreementCapacity,Error>{
        reserve(&self.funding,&[size_of::<&Self>(),size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let actual=self.body().owner.borrow()?;
        agreement_capacity(&actual,&self.body().inputs)
    }
}
impl OriginalCommunicationOwner {
    /// Consumes the exact initialized communicator table. No tensor-group
    /// selector, mutable model state or family geometry is needed or fabricated.
    pub(crate) fn prepare_control_source(self,group:CollectiveGroupId,
        pool:&eredu_runtime::working_memory::WorkingMemoryPool)->Result<OriginalParallelControlSource,Error>{
        reserve(self.funding(),&[size_of::<Self>(),size_of::<Body>(),
            size_of::<OriginalParallelControlSource>(),size_of::<Option<Rc<Body>>>(),
            size_of::<Result<OriginalParallelControlSource,Error>>(),
            size_of::<(&Self,CollectiveGroupId,&eredu_runtime::working_memory::WorkingMemoryPool)>(),
            size_of::<agreement::OriginalAgreementInputs>(),size_of::<AgreementCapacity>(),
            Layout::new::<[usize;2]>().extend(Layout::new::<Body>()).map_err(|_|overflow())?.0.pad_to_align().size(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?])?;
        let self_=self.pin_registered_buffers(pool)?;
        let inputs={
            let actual=self_.borrow()?;
            let selected=actual.source().manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
                .map_err(|cause|failure(Cause::Rank(cause),actual.source(),self_.funding()))?;
            let native=actual.group(selected.order()).ok_or_else(||failure(Cause::Resource,actual.source(),self_.funding()))?.0;
            if native.has_original_parallel() || native.has_original_control()
                || native.original_control_request().is_some() || native.retained_transport_stream().is_none() {
                return Err(failure(Cause::Identity,actual.source(),self_.funding()));
            }
            let inputs=actual.prepare_agreement_inputs(group,pool)?;
            // The shared operation compiler enforces the actual native source,
            // persistent census, physical group and exact completed status leaf.
            agreement_capacity(&actual,&inputs)?;
            inputs
        };
        let retained=self_.source().clone();let funding=self_.funding().clone();
        Ok(OriginalParallelControlSource{body:Some(Rc::new(Body{
            inputs,owner:self_,taken:Cell::new(false),retained,funding:funding.clone(),
        })),funding})
    }
}

/// Both exact source kinds feed the one existing cursor and native role worker.
/// The control variant grants no tensor-source or model-invocation capability.
#[derive(Clone)]
pub(super) enum ControlSource {
    Model(OriginalParallelSource),
    Control(OriginalParallelControlSource),
}
impl ControlSource {
    pub(super) fn funding(&self)->&WorkspaceMetadataFunding{match self{
        Self::Model(source)=>source.funding(),Self::Control(source)=>&source.funding,
    }}
    pub(super) fn communication_source(&self)->Result<OriginalCommunicationSource<'_>,Error>{match self{
        Self::Model(source)=>source.communication_source(),Self::Control(source)=>source.body().owner.borrow(),
    }}
    pub(super) fn agreement_inputs(&self)->Option<&agreement::OriginalAgreementInputs>{match self{
        Self::Model(source)=>source.agreement_inputs(),Self::Control(source)=>Some(&source.body().inputs),
    }}
    pub(super) fn model(&self)->Option<&OriginalParallelSource>{match self{Self::Model(source)=>Some(source),Self::Control(_)=>None}}
    pub(super) fn take_control_owner(&self)->Result<(),Error>{match self{
        Self::Model(source)=>source.take_control_owner(),
        Self::Control(source)=>{
            reserve(&source.funding,&[size_of::<&Self>(),size_of::<Result<(),Error>>(),
                failure_control_bytes().ok_or_else(overflow)?])?;
            if source.body().taken.replace(true){
                return Err(control_error(ControlCause::Exhausted,&source.body().retained,&source.funding));
            }
            Ok(())
        },
    }}
    pub(super) fn control_capacity(&self)->Result<AgreementCapacity,Error>{
        reserve(self.funding(),&[size_of::<&Self>(),size_of::<AgreementCapacity>(),
            size_of::<Result<AgreementCapacity,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let source=self.communication_source()?;
        let inputs=self.agreement_inputs().ok_or_else(||failure(Cause::Resource,source.source(),self.funding()))?;
        match self {
            Self::Model(_)=>model_agreement_capacity(&source,inputs),
            Self::Control(_)=>agreement_capacity(&source,inputs),
        }
    }
}
pub(super) fn agreement_capacity(source:&OriginalCommunicationSource<'_>,inputs:&agreement::OriginalAgreementInputs)
    ->Result<AgreementCapacity,Error>{
    reserve(source.funding(),&[size_of::<(&OriginalCommunicationSource<'_>,&agreement::OriginalAgreementInputs)>(),
        size_of::<[AgreementCapacity;2]>(),size_of::<Result<AgreementCapacity,Error>>()])?;
    Ok(inputs.capacity(source,false)?.union(inputs.capacity(source,true)?))
}

/// Actual immutable status leaves for one explicitly selected initialized group.
/// The scalar inputs and ordinary worker are shared; peer geometry is not.
pub(super) fn agreement_capacity_for_group(source:&OriginalCommunicationSource<'_>,
    inputs:&agreement::OriginalAgreementInputs,group:CollectiveGroupId)->Result<AgreementCapacity,Error>{
    reserve(source.funding(),&[size_of::<(&OriginalCommunicationSource<'_>,
        &agreement::OriginalAgreementInputs,CollectiveGroupId)>(),
        size_of::<[AgreementCapacity;2]>(),size_of::<Result<AgreementCapacity,Error>>()])?;
    Ok(inputs.capacity_for_group(source,group,false)?.union(inputs.capacity_for_group(source,group,true)?))
}
/// A model request can reach its local tensor provider group as well as its
/// session group. Price the finite retained manifest's actual agreement leaves,
/// before any occurrence is admitted; this does not grant an occurrence.
pub(super) fn model_agreement_capacity(source:&OriginalCommunicationSource<'_>,
    inputs:&agreement::OriginalAgreementInputs)->Result<AgreementCapacity,Error>{
    reserve(source.funding(),&[size_of::<(&OriginalCommunicationSource<'_>,
        &agreement::OriginalAgreementInputs)>(),size_of::<AgreementCapacity>(),
        size_of::<std::slice::Iter<'_,CommunicationGroupDescriptor>>(),
        size_of::<std::slice::Iter<'_,eredu_runtime::CommunicationOperationRequirement>>(),
        size_of::<Result<AgreementCapacity,Error>>()])?;
    let mut capacity=agreement_capacity(source,inputs)?;
    for group in source.source().manifest().groups() {
        if group.id()==inputs.group() || group.local_index().is_none()
            || !group.requirements().operations().iter().any(|operation|
                operation.operation()==CommunicationOperation::FailureAgreement) {continue;}
        capacity=capacity.union(agreement_capacity_for_group(source,inputs,group.id())?);
    }
    Ok(capacity)
}
