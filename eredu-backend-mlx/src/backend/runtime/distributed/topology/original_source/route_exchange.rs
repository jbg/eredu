//! Exact route/endpoint source for the existing logical pair exchange.
use super::*;
use crate::backend::runtime::distributed::group::LogicalExchangePlan;
use safemlx::distributed::{GroupCpuExchangeStorage, GroupWorkerOperation};
use safemlx::{OriginalScopeObserver,PreparedInputRuntime};
use crate::backend::runtime::distributed::completion::{OriginalCommunicationCompletion,
    prepared::{CompletionResourceLayout,PreparedCompletionResources,ReadyCompletionResources}};

/// Borrowed actual route and the same group plan used by ordinary execution.
/// A query cannot replace its peers, round population or participation proof.
pub(crate) struct OriginalRouteExchange<'a> {
    route: &'a CommunicationRouteRealization,
    plan: LogicalExchangePlan<'a>,
    order: usize,
    source: RetainedCommunicationSource,
    funding: WorkspaceMetadataFunding,
}

// The directed selected route, not native rank numbering, selects complementary
// blocking socket order. The actual pair has one CPU stream on each endpoint.
#[derive(Clone,Copy)]
enum SubmissionOrder { SendFirst, ReceiveFirst }
impl SubmissionOrder {
    fn reverses_roots(self)->bool {matches!(self,Self::SendFirst)}
}

/// The selected neutral resource owns its own completion entry. A logical
/// collective cannot borrow a route ID or masquerade as a native world sum.
#[derive(Clone, Copy)]
enum ExchangeSelection { Route(usize), Group(usize) }
impl ExchangeSelection {
    fn group<'a>(self,source:&'a OriginalCommunicationSource<'_>)->Result<&'a Group,Error>{
        source.funding.reserve_metadata(size_of::<(Self,&OriginalCommunicationSource<'_>,
            Option<&Group>,Result<&Group,Error>)>()).map_err(Error::WorkspacePlanning)?;
        let group=match self {
            Self::Route(order)=>source.route(order).and_then(|(route,_,_)|
                source.matches_route(order,route).then(||route.group()).flatten()),
            Self::Group(order)=>source.group(order).and_then(|(group,_,_)|
                source.matches_group(order,group).then_some(group)),
        };
        group.ok_or_else(||failure(Cause::Identity,&source.source,&source.funding))
    }
    fn persistent<'a>(self,source:&'a OriginalCommunicationSource<'_>)
        ->Result<OriginalCommunicatorPersistent<'a>,Error>{
        source.funding.reserve_metadata(size_of::<(Self,&OriginalCommunicationSource<'_>,
            Result<OriginalCommunicatorPersistent<'_>,Error>)>()).map_err(Error::WorkspacePlanning)?;
        match self {
            Self::Route(order)=>source.route_persistent(order)?
                .ok_or_else(||failure(Cause::Resource,&source.source,&source.funding)),
            Self::Group(order)=>source.group_persistent(order),
        }
    }
    fn completion(self,source:&OriginalCommunicationSource<'_>,traversal:safemlx::OperationEvalTraversalLayout)
        ->Result<ReadyCompletionResources,Error>{
        source.funding.reserve_metadata(size_of::<(Self,&OriginalCommunicationSource<'_>,
            safemlx::OperationEvalTraversalLayout,(usize,usize),PreparedCompletionResources<'_,'_>,
            Result<ReadyCompletionResources,Error>)>()).map_err(Error::WorkspacePlanning)?;
        let (groups,routes)=match self {Self::Route(_)=>(0,1),Self::Group(_)=>(1,0)};
        let mut prepared=PreparedCompletionResources::prepare_original(source,
            CompletionResourceLayout{arrays:0,counts:&[],groups,routes,streams:0},traversal)?;
        match self {Self::Route(order)=>prepared.retain_route(order)?,Self::Group(order)=>prepared.retain_group(order)?}
        prepared.finish()
    }
}

/// Two actual native operations and their one completion for a selected round.
/// Arithmetic, framing and the enclosing role remain separate producers.
/// Each value keeps its actual persistent source and H, including failed exits.
pub(crate) struct OriginalRouteRound<'a> {
    native: GroupCpuExchangeStorage<'a>,
    _persistent: RoundPersistent<'a>,
    selection: ExchangeSelection,
    round: usize,
    submission_order: SubmissionOrder,
    source: RetainedCommunicationSource,
    funding: WorkspaceMetadataFunding,
}
enum RoundPersistent<'a> {
    Queried([OriginalCommunicatorPersistent<'a>;2]),
    // The owned cold producer retains the exact native table and original
    // persistent-source qualification throughout this actual-input loan.
    Retained(&'a super::parallel::OriginalParallelSource),
}
impl OriginalCommunicationSource<'_> {
    /// A nonlocal route has no native endpoint; the ordinary shared wave driver
    /// owns that branch. An endpoint needing packed world transport refuses
    /// until that distinct ordinary equation has a complete original producer.
    pub(crate) fn route_exchange(&self, order: usize)
        -> Result<Option<OriginalRouteExchange<'_>>, Error>
    {
        let value=self.framed_route_exchange(order)?;
        if value.as_ref().is_some_and(|value|value.rounds()!=1){
            return Err(failure(Cause::LogicalWorldTransport,&self.source,&self.funding));
        }
        Ok(value)
    }
    /// Exact ordinary forwarding path. A multi-round path is available only
    /// under the retained setup proof that every native world rank participates
    /// in this shared route wave; each round still needs its own native source.
    pub(crate) fn framed_route_exchange(&self, order: usize)
        -> Result<Option<OriginalRouteExchange<'_>>, Error>
    {
        let controls = [size_of::<OriginalRouteExchange<'_>>(),
            size_of::<Result<Option<OriginalRouteExchange<'_>>, Error>>(),
            size_of::<Option<(&CommunicationRouteRealization, &CommunicationRouteDescriptor, bool)>>(),
            size_of::<Result<Option<LogicalExchangePlan<'_>>, crate::backend::runtime::distributed::group::LogicalExchangeCause>>(),
            size_of::<(&Self, usize)>(), size_of::<(usize,usize,usize)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.validate()?;
        let (route,_,_) = self.route(order).ok_or_else(|| failure(Cause::Resource,&self.source,&self.funding))?;
        if !self.matches_route(order,route) { return Err(failure(Cause::Identity,&self.source,&self.funding)); }
        let Some(group) = route.group() else { return Ok(None); };
        let plan = group.logical_exchange_plan()
            .map_err(|cause| failure(Cause::LogicalExchange(cause),&self.source,&self.funding))?
            .ok_or_else(|| failure(Cause::LogicalWorldTransport,&self.source,&self.funding))?;
        if plan.rounds()>1 && !self.source.route(order).is_some_and(|(_,world_wave)|world_wave) {
            return Err(failure(Cause::LogicalWorldTransport,&self.source,&self.funding));
        }
        Ok(Some(OriginalRouteExchange { route,plan,order,source:self.source.clone(),funding:self.funding.clone() }))
    }
}
impl OriginalRouteExchange<'_> {
    pub(crate) fn rounds(&self) -> usize { self.plan.rounds() }
    pub(crate) fn route(&self) -> &CommunicationRouteRealization { self.route }
    pub(crate) fn source(&self) -> &RetainedCommunicationSource { &self.source }

    /// Native edge identity comes from the actual selected group plan, never
    /// from caller-provided operation codes or equal peer geometry. The next
    /// round receives its actual settled predecessor; this pure source query
    /// does not evaluate, clone, detach or publish an input.
    pub(crate) fn round_storage<'a>(&'a self, source:&'a OriginalCommunicationSource<'_>,
        round:usize,input:&'a Array) -> Result<OriginalRouteRound<'a>,Error>
    {
        let parts=[size_of::<OriginalRouteRound<'a>>(),size_of::<Result<OriginalRouteRound<'a>,Error>>(),
            size_of::<OriginalRouteLayoutRound<'a>>(),size_of::<(&Self,&OriginalCommunicationSource<'_>,usize,&Array)>(),
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        self.round_layout_storage(source,round,input.shape(),input.dtype())?.bind_actual(source,input)
    }
}
/// Same selected pair plus finite route/recovery/housekeeping destinations,
/// prepared before a native constructor can accept either edge.
pub(crate) struct PreparedRouteRound<'a> {
    round: OriginalRouteRound<'a>,
    ready: ReadyCompletionResources,
    backing_capacity: usize,
}
impl OriginalRouteRound<'_> {
    pub(crate) fn graph_capacity(&self)->usize{self.native.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.native.record_capacity()}
    pub(crate) fn round(&self) -> usize { self.round }
    pub(crate) fn source(&self) -> &RetainedCommunicationSource { &self.source }
}
impl<'a> OriginalRouteRound<'a> {
    pub(crate) fn backing_capacity(&self,source:&OriginalCommunicationSource<'_>,runtime:&PreparedInputRuntime)
        ->Result<usize,Error> {
        let controls=[size_of::<Self>(),size_of::<usize>(),
            size_of::<Result<usize,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&PreparedInputRuntime)>(),
            size_of::<(usize,usize)>(),size_of::<Option<usize>>(),
            self.native.send().backing_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            self.native.receive().backing_storage_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.source.same_source(source.source()){return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let send=self.native.send().backing_storage(runtime)
            .map_err(|cause|failure(Cause::Buffer(cause),&self.source,&self.funding))?.capacity();
        let receive=self.native.receive().backing_storage(runtime)
            .map_err(|cause|failure(Cause::Buffer(cause),&self.source,&self.funding))?.capacity();
        let backing_capacity=send.checked_add(receive)
            .ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(backing_capacity)
    }
    pub(crate) fn prepare(self,source:&OriginalCommunicationSource<'_>,runtime:&PreparedInputRuntime)
        ->Result<PreparedRouteRound<'a>,Error>
    {
        self.funding.reserve_metadata([size_of::<Self>(),size_of::<PreparedRouteRound<'a>>(),
            size_of::<Result<PreparedRouteRound<'a>,Error>>()].into_iter().try_fold(size_of::<[usize;3]>(),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let backing_capacity=self.backing_capacity(source,runtime)?;
        let ready=self.selection.completion(source,self.native.traversal())?;
        Ok(PreparedRouteRound{round:self,ready,backing_capacity})
    }
}
/// Both actual lazy roots plus the source that paid their constructors. A
/// native event and the enclosing Q retain them until completion or recovery.
pub(crate) struct ConstructedRouteExchange {
    outputs:[Array;2],
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl ConstructedRouteExchange {
    pub(crate) fn outputs(&self)->&[Array;2]{&self.outputs}
    pub(crate) fn into_parts(self)->([Array;2],RetainedCommunicationSource,WorkspaceMetadataFunding){
        (self.outputs,self.source,self.funding)
    }
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
}
/// Only the paired constructor can mint this accepted source. The final event
/// does not consult fresh availability after the first edge has been accepted.
pub(crate) struct AcceptedRouteSource<'stream> {
    value:ConstructedRouteExchange,
    observer:OriginalScopeObserver,
    stream:&'stream Stream,
    traversal:safemlx::OperationEvalTraversalLayout,
    submission_order:SubmissionOrder,
    funding:WorkspaceMetadataFunding,
}
pub(crate) struct AcceptedRouteExchange<'stream> {
    source:AcceptedRouteSource<'stream>,
    ready:ReadyCompletionResources,
}
impl PreparedRouteRound<'_> {
    pub(crate) fn graph_capacity(&self)->usize{self.round.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.round.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize{self.backing_capacity}
    pub(crate) fn construct_accepted<'stream>(self,source:&OriginalCommunicationSource<'_>,
        observer:&OriginalScopeObserver,stream:&'stream Stream)->Result<AcceptedRouteExchange<'stream>,Error>
    {
        let controls=[size_of::<Self>(),size_of::<ConstructedRouteExchange>(),
            size_of::<AcceptedRouteSource<'stream>>(),size_of::<AcceptedRouteExchange<'stream>>(),
            size_of::<Result<AcceptedRouteExchange<'stream>,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&OriginalScopeObserver,&Stream)>(),
            size_of::<[Array;2]>(),size_of::<Result<[Array;2],safemlx::error::Exception>>(),
            self.round.native.construction_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            safemlx::OriginalScopeObserver::control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        self.round.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.round.source.same_source(source.source()){
            return Err(failure(Cause::Identity,&self.round.source,&self.round.funding));
        }
        let traversal=self.round.native.traversal();
        let outputs=self.round.native.construct_original(observer,stream)
            .map_err(|cause|failure(Cause::Native(cause),&self.round.source,&self.round.funding))?;
        let value=ConstructedRouteExchange{outputs,source:self.round.source,funding:self.round.funding.clone()};
        Ok(AcceptedRouteExchange{source:AcceptedRouteSource{value,observer:observer.clone(),stream,
            traversal,submission_order:self.round.submission_order,funding:self.round.funding},ready:self.ready})
    }
}
impl AcceptedRouteSource<'_> {
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.value.source}
    pub(crate) fn observer(&self)->&OriginalScopeObserver{&self.observer}
    pub(crate) fn stream(&self)->&Stream{self.stream}
    pub(crate) fn traversal(&self)->safemlx::OperationEvalTraversalLayout{self.traversal}
    pub(crate) fn outputs(&self)->&[Array]{&self.value.outputs}
}
impl AcceptedRouteExchange<'_> {
    pub(crate) fn submit(mut self)->Result<(ConstructedRouteExchange,OriginalCommunicationCompletion),Error>{
        let controls=[size_of::<Self>(),size_of::<Result<(ConstructedRouteExchange,OriginalCommunicationCompletion),Error>>()];
        self.source.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        // MLX appends both independent outputs to its BFS tape in root order
        // and dispatches that tape in reverse. Source therefore submits
        // [receive,send], Destination [send,receive]. These exact two operations
        // and roots were already quoted; order adds no tensor or task. Restore
        // public [send,receive] even after a rejected submission, with no clone.
        let reverse=self.source.submission_order.reverses_roots();
        if reverse {self.source.value.outputs.swap(0,1);}
        let completion=self.ready.submit_accepted_route(&self.source);
        if reverse {self.source.value.outputs.swap(0,1);}
        let completion=completion?;
        Ok((self.source.value,completion))
    }
}

mod layout;
pub(crate) use layout::{OriginalRouteLayoutRound,OwnedOriginalRouteLayoutRound,OwnedOriginalExchangeLayoutRound};

mod group;
pub(crate) use group::OriginalGroupExchange;
