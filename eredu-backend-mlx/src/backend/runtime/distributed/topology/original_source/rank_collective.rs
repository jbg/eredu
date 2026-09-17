//! Exact selected-rank consumer of the existing native operation/completion worker.
use super::*;
use super::operation_storage::OriginalCommunicationCompletedOperation;
use crate::backend::runtime::distributed::completion::{OriginalCommunicationCompletion,
    prepared::ReadyCompletionResources};
use safemlx::{OriginalScopeObserver, PreparedInputRuntime, distributed::GroupWorkerOperation};

/// Semantic equation supported by this native producer. Other gather axes and
/// logical subgroup/world-wave equations keep their own unqualified producers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OriginalRankCollective { Sum, GatherAxisZero }
impl OriginalRankCollective {
    fn operation(self)->CommunicationOperation {match self {
        Self::Sum=>CommunicationOperation::AllReduceSum,
        Self::GatherAxisZero=>CommunicationOperation::AllGatherEven,
    }}
    fn native(self)->GroupWorkerOperation {match self {
        Self::Sum=>GroupWorkerOperation::Sum,Self::GatherAxisZero=>GroupWorkerOperation::Gather,
    }}
}

/// Exact selected ID/member/operation plus completed native input and finite
/// completion destinations, all prepared before original graph construction.
/// These facts are not an admission grant; the enclosing role still owns its
/// actual Graph/Record/backing reservations and submission authority.
pub(crate) struct PreparedRankCollective<'a> {
    operation:OriginalCommunicationCompletedOperation<'a>,
    ready:ReadyCompletionResources,
    input:&'a Array,
    runtime:&'a PreparedInputRuntime,
    collective:OriginalRankCollective,
    members:usize,
    backing_capacity:usize,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
/// Owns the actual accepted graph and the matching preallocated completion.
/// Consuming it never asks whether a *new* submission would still be admitted.
pub(crate) struct AcceptedRankCollective<'stream> {
    accepted:AcceptedCommunicationSource<'stream>,
    ready:ReadyCompletionResources,
    _funding:WorkspaceMetadataFunding,
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn prepare_rank_collective<'a>(&'a self,id:CollectiveGroupId,input:&'a Array,
        collective:OriginalRankCollective,runtime:&'a PreparedInputRuntime)
        ->Result<PreparedRankCollective<'a>,Error> {
        let parts=[size_of::<PreparedRankCollective<'a>>(),
            size_of::<Result<PreparedRankCollective<'a>,Error>>(),
            size_of::<(&Self,CollectiveGroupId,&Array,OriginalRankCollective,&PreparedInputRuntime)>(),
            size_of::<eredu_runtime::CommunicationGroupOperation<'_>>(),
            size_of::<Result<eredu_runtime::CommunicationGroupOperation<'_>,eredu_runtime::CommunicationGroupOperationError>>(),
            size_of::<Result<(),eredu_runtime::CommunicationTensorContractError>>(),
            size_of::<TensorDtype>(),size_of::<(usize,usize)>(),size_of::<Option<usize>>(),
            eredu_runtime::CommunicationManifest::group_operation_control_bytes()
                .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            eredu_runtime::CommunicationOperationRequirement::tensor_metadata_control_bytes()
                .and_then(|bytes|bytes.checked_mul(2)).ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            size_of::<&Self>().checked_mul(3).ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        let selected=self.source.manifest().select_group_operation(id,collective.operation())
            .map_err(|cause|failure(Cause::Rank(cause),&self.source,&self.funding))?;
        let requirement=selected.requirement();
        if !requirement.exact_completion() || (collective==OriginalRankCollective::GatherAxisZero && input.ndim()==0) {
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        let dtype=crate::tensor::portable_dtype(input.dtype());
        requirement.validate_tensor_metadata(&dtype,input.ndim(),Some(input.size()),false)
            .map_err(|cause|failure(Cause::Tensor(cause),&self.source,&self.funding))?;
        // The native source checks exact actual group/input identity and keeps
        // logical subgroups explicit. It never substitutes the native world for
        // a selected pack/exchange/reduce/unpack equation.
        let operation=self.group_cpu_operation_storage(selected.order(),input,collective.native())?;
        let (rank,elements)=operation.native().constructor().output_geometry();
        requirement.validate_tensor_metadata(&dtype,rank,Some(elements),true)
            .map_err(|cause|failure(Cause::Tensor(cause),&self.source,&self.funding))?;
        let backing_capacity=operation.backing_storage(runtime)?.capacity();
        let operation=operation.with_completion()?;
        let ready=operation.prepare_resources(self,Some(selected.order()))?;
        Ok(PreparedRankCollective{operation,ready,input,runtime,collective,
            members:selected.descriptor().members().len(),backing_capacity,
            source:self.source.clone(),funding:self.funding.clone()})
    }
}
impl PreparedRankCollective<'_> {
    pub(crate) fn graph_capacity(&self)->usize {self.operation.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize {self.operation.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize {self.backing_capacity}
    pub(crate) fn runtime(&self)->&PreparedInputRuntime {self.runtime}
    pub(crate) fn construct_accepted<'stream>(self,source:&OriginalCommunicationSource<'_>,
        observer:&OriginalScopeObserver,stream:&'stream Stream)->Result<AcceptedRankCollective<'stream>,Error> {
        let parts=[size_of::<Self>(),size_of::<AcceptedRankCollective<'stream>>(),
            size_of::<Result<AcceptedRankCollective<'stream>,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_,i32>>>(),
            size_of::<(usize,&i32)>(),size_of::<Option<usize>>(),size_of::<bool>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let accepted=self.operation.construct_accepted(source,observer,stream)?;
        let output=&accepted.outputs()[0];
        // Actual lazy metadata is immutable through this closed carrier. Check
        // the equation before the accepted event can submit any native task.
        let exact=output.dtype()==self.input.dtype() && output.ndim()==self.input.ndim()
            && self.input.shape().iter().enumerate().all(|(axis,&dimension)| {
                let expected=usize::try_from(dimension).ok().and_then(|dimension|
                    if self.collective==OriginalRankCollective::GatherAxisZero && axis==0 {
                        dimension.checked_mul(self.members)
                    } else {Some(dimension)});
                expected==usize::try_from(output.shape()[axis]).ok()
            });
        if !exact {return Err(failure(Cause::Output,&self.source,&self.funding));}
        Ok(AcceptedRankCollective{accepted,ready:self.ready,_funding:self.funding})
    }
}
impl AcceptedRankCollective<'_> {
    pub(crate) fn submit(self)->Result<(OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error> {
        let parts=[size_of::<Self>(),size_of::<Result<(OriginalCommunicationConstructed,OriginalCommunicationCompletion),Error>>()];
        self._funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.accepted.submit(self.ready)
    }
}
