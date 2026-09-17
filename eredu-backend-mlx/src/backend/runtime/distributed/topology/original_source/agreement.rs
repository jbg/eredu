//! The ordinary payload-free I32 vote using retained original source workers.
use super::*;
use super::operation_storage::OriginalCommunicationCompletedOperation;
use crate::backend::runtime::distributed::completion::{
    prepared::ReadyCompletionResources,PreparedCommunicationScalar,OriginalCommunicationBool,
    MlxNeuralCommunicationCompletion,
};
use safemlx::{PreparedInputRuntime,OriginalScopeObserver,distributed::GroupWorkerOperation};
mod inputs;
mod chain;
pub(crate) use inputs::OriginalAgreementInputs;

/// Finite native operation, scalar destination and existing completion prepared
/// from the actual selected status leaf. No Graph or Record grant is issued here.
struct PreparedNativeAgreement<'a> {
    operation:OriginalCommunicationCompletedOperation<'a>,
    scalar:PreparedCommunicationScalar,
    ready:ReadyCompletionResources,
    runtime:&'a PreparedInputRuntime,
    backing:usize,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
pub(crate) struct PreparedOriginalAgreement<'a>{kind:PreparedAgreementKind<'a>}
enum PreparedAgreementKind<'a>{Native(PreparedNativeAgreement<'a>),Chain(chain::PreparedStatusAgreement<'a>)}
impl PreparedOriginalAgreement<'_>{
    pub(crate) fn graph_capacity(&self)->usize{match &self.kind{PreparedAgreementKind::Native(value)=>value.graph_capacity(),PreparedAgreementKind::Chain(value)=>value.capacity().graph}}
    pub(crate) fn record_capacity(&self)->usize{match &self.kind{PreparedAgreementKind::Native(value)=>value.record_capacity(),PreparedAgreementKind::Chain(value)=>value.capacity().records}}
    pub(crate) fn backing_capacity(&self)->usize{match &self.kind{PreparedAgreementKind::Native(value)=>value.backing_capacity(),PreparedAgreementKind::Chain(value)=>value.capacity().backing}}
    pub(crate) fn runtime(&self)->&PreparedInputRuntime{match &self.kind{PreparedAgreementKind::Native(value)=>value.runtime(),PreparedAgreementKind::Chain(value)=>value.runtime()}}
    pub(crate) fn submit(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<(OriginalCommunicationBool,MlxNeuralCommunicationCompletion),Error>{
        match self.kind{PreparedAgreementKind::Native(value)=>value.submit(source,observer,stream),PreparedAgreementKind::Chain(value)=>value.submit(source,observer,stream)}
    }
}
/// Maximum of the two actual immutable protocol branches, for one vote.
/// This grants neither an occurrence nor a native allocation domain.
#[derive(Clone,Copy,Debug)]
pub(crate) struct AgreementCapacity {
    pub(crate) graph:usize,
    pub(crate) records:usize,
    pub(crate) backing:usize,
}
impl AgreementCapacity {
    pub(crate) fn union(self,other:Self)->Self {
        Self{graph:self.graph.max(other.graph),records:self.records.max(other.records),
            backing:self.backing.max(other.backing)}
    }
    pub(crate) fn covers(self,actual:Self)->bool {
        actual.graph<=self.graph && actual.records<=self.records && actual.backing<=self.backing
    }
}
impl OriginalAgreementInputs {
    pub(crate) fn prepare<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,success:bool)
        ->Result<PreparedOriginalAgreement<'a>,Error> {
        self.prepare_for_group(source,self.group(),success)
    }
    pub(crate) fn prepare_for_group<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,
        group:CollectiveGroupId,success:bool)->Result<PreparedOriginalAgreement<'a>,Error> {
        reserve(source.funding(),&[
            size_of::<PreparedOriginalAgreement<'a>>(),
            size_of::<Result<PreparedOriginalAgreement<'a>,Error>>(),
            size_of::<(&Self,&OriginalCommunicationSource<'_>,CollectiveGroupId,bool)>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        if !self.same_source(source){return Err(failure(Cause::Identity,source.source(),source.funding()));}source.validate()?;
        let selected=source.source().manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),source.source(),source.funding()))?;
        if !selected.requirement().exact_completion(){return Err(failure(Cause::Resource,source.source(),source.funding()));}
        if let Some(value)=chain::PreparedStatusAgreement::prepare(self,source,selected.order(),success)?{return Ok(PreparedOriginalAgreement{kind:PreparedAgreementKind::Chain(value)});}
        let (operation,backing)=self.prepare_operation(source,group,success)?;
        let selected=source.source().manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),source.source(),source.funding()))?;
        let ready=operation.prepare_resources(source,Some(selected.order()))?;
        let scalar=PreparedCommunicationScalar::prepare_agreement(source,selected.order())?;
        Ok(PreparedOriginalAgreement{kind:PreparedAgreementKind::Native(PreparedNativeAgreement{operation,scalar,ready,runtime:self.runtime(),backing,
            source:source.source().clone(),funding:source.funding().clone()})})
    }
}
impl OriginalAgreementInputs {
    fn prepare_operation<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,group:CollectiveGroupId,success:bool)
        ->Result<(OriginalCommunicationCompletedOperation<'a>,usize),Error> {
        reserve(source.funding(),&[
            size_of::<(&Self,&OriginalCommunicationSource<'_>,CollectiveGroupId,bool)>(),
            size_of::<Result<(OriginalCommunicationCompletedOperation<'a>,usize),Error>>(),
            CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        if !self.same_source(source) {return Err(failure(Cause::Identity,source.source(),source.funding()));}
        source.validate()?;
        let selected=source.source().manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),source.source(),source.funding()))?;
        if !selected.requirement().exact_completion() {
            return Err(failure(Cause::Resource,source.source(),source.funding()));
        }
        // The protocol itself creates a single private I32 status word. The
        // payload-free declaration grants no logical tensor or dtype allowance.
        let input=self.value(success);
        let operation=source.group_cpu_operation_storage(selected.order(),input,GroupWorkerOperation::Sum)?;
        if operation.native().constructor().output_geometry()!=(1,1) {
            return Err(failure(Cause::Output,source.source(),source.funding()));
        }
        let backing=operation.backing_storage(self.runtime())?.capacity();
        let operation=operation.with_completion()?;
        Ok((operation,backing))
    }
    pub(crate) fn capacity(&self,source:&OriginalCommunicationSource<'_>,success:bool)
        ->Result<AgreementCapacity,Error> {
        self.capacity_for_group(source,self.group(),success)
    }
    pub(crate) fn capacity_for_group(&self,source:&OriginalCommunicationSource<'_>,
        group:CollectiveGroupId,success:bool)->Result<AgreementCapacity,Error> {
        reserve(source.funding(),&[size_of::<AgreementCapacity>(),size_of::<Result<AgreementCapacity,Error>>(),
            size_of::<(&Self,&OriginalCommunicationSource<'_>,CollectiveGroupId,bool)>(),
            size_of::<(OriginalCommunicationCompletedOperation<'_>,usize)>()])?;
        if !self.same_source(source){return Err(failure(Cause::Identity,source.source(),source.funding()));}source.validate()?;
        let selected=source.source().manifest().select_group_operation(group,CommunicationOperation::FailureAgreement)
            .map_err(|cause|failure(Cause::Rank(cause),source.source(),source.funding()))?;
        if !selected.requirement().exact_completion(){return Err(failure(Cause::Resource,source.source(),source.funding()));}
        if let Some(value)=chain::PreparedStatusAgreement::prepare(self,source,selected.order(),success)?{return Ok(value.capacity());}
        let (operation,backing)=self.prepare_operation(source,group,success)?;
        Ok(AgreementCapacity{graph:operation.graph_capacity(),records:operation.record_capacity(),backing})
    }
}
impl PreparedNativeAgreement<'_> {
    pub(crate) fn graph_capacity(&self)->usize {self.operation.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize {self.operation.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize {self.backing}
    pub(crate) fn runtime(&self)->&PreparedInputRuntime {self.runtime}
    /// The enclosing phase supplies its actual role and native stream. The
    /// original accepted handoff is never re-admitted after construction.
    pub(crate) fn submit(self,source:&OriginalCommunicationSource<'_>,observer:&OriginalScopeObserver,stream:&Stream)
        ->Result<(OriginalCommunicationBool,MlxNeuralCommunicationCompletion),Error> {
        reserve(&self.funding,&[
            size_of::<Self>(),size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            size_of::<Result<(OriginalCommunicationBool,MlxNeuralCommunicationCompletion),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let accepted=self.operation.construct_accepted(source,observer,stream)?;
        let (result,completion)=self.scalar.submit_accepted(accepted,self.ready)?;
        Ok((result,completion.into()))
    }
}
fn overflow()->Error {Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)}
fn reserve(funding:&WorkspaceMetadataFunding,bytes:&[usize])->Result<(),Error> {
    funding.reserve_metadata(bytes.iter().copied().try_fold(std::mem::size_of_val(bytes),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
