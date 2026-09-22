//! The actual logical pair first names its complete cold source, then binds
//! the real settled frame before the unchanged accepted constructor/completion.
use super::*;
use safemlx::distributed::{GroupCpuLayoutStorage,GroupCpuExchangeLayoutStorage};

pub(crate) struct OriginalRouteLayoutRound<'a> {
    native: GroupCpuExchangeLayoutStorage<'a>,
    _persistent: [OriginalCommunicatorPersistent<'a>;2],
    selection: ExchangeSelection,
    round: usize,
    submission_order: SubmissionOrder,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
impl OriginalRouteExchange<'_> {
    /// Queries the exact existing route peers and immutable frame layout. No
    /// placeholder native tensor or storage capacity authority is created.
    pub(crate) fn round_layout_storage<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,
        round:usize,shape:&'a [i32],dtype:safemlx::Dtype)->Result<OriginalRouteLayoutRound<'a>,Error> {
        let submission_order=match self.route.endpoint() {
            Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Source)=>SubmissionOrder::SendFirst,
            Some(crate::backend::runtime::distributed::topology::CommunicationRouteEndpoint::Destination)=>SubmissionOrder::ReceiveFirst,
            None=>return Err(failure(Cause::Identity,&self.source,&self.funding)),
        };
        OriginalRouteLayoutRound::query(source,&self.source,&self.funding,self.plan,
            ExchangeSelection::Route(self.order),submission_order,round,shape,dtype)
    }
}
impl<'a> OriginalRouteLayoutRound<'a> {
    pub(super) fn query(source:&'a OriginalCommunicationSource<'_>,retained:&RetainedCommunicationSource,
        funding:&HostMetadataFunding,plan:LogicalExchangePlan<'a>,selection:ExchangeSelection,
        submission_order:SubmissionOrder,round:usize,shape:&'a [i32],dtype:safemlx::Dtype)
        ->Result<Self,Error>{
        Self::query_asymmetric(source, retained, funding, plan, selection, submission_order,
            round, shape, shape, dtype)
    }
    pub(super) fn query_asymmetric(source: &'a OriginalCommunicationSource<'_>, retained: &RetainedCommunicationSource,
        funding: &HostMetadataFunding, plan: LogicalExchangePlan<'a>, selection: ExchangeSelection,
        submission_order: SubmissionOrder, round: usize, shape: &'a [i32], receive_shape: &'a [i32], dtype: safemlx::Dtype)
        -> Result<Self, Error> {
        let group=plan.group();
        let overflow=||Error::WorkspacePlanning(HostMetadataFundingError::Overflow);
        let controls=[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&RetainedCommunicationSource,&HostMetadataFunding,
                LogicalExchangePlan<'_>,ExchangeSelection,SubmissionOrder,usize,&[i32],&[i32],safemlx::Dtype)>(),
            size_of::<(usize,usize)>(),size_of::<(i32,i32)>(),
            size_of::<[OriginalCommunicatorPersistent<'_>;2]>(),
            size_of::<[GroupCpuLayoutStorage<'_>;2]>(),
            size_of::<Result<GroupCpuLayoutStorage<'_>,safemlx::distributed::GroupStorageUnavailable>>(),
            size_of::<Result<&Group,Error>>(),size_of::<Result<OriginalCommunicatorPersistent<'_>,Error>>(),
            failure_control_bytes().ok_or_else(overflow)?,
            group.native_group().cpu_layout_storage_control_bytes().and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !retained.same_source(source.source()) || !std::ptr::eq(selection.group(source)?,group) {
            return Err(failure(Cause::Identity,retained,funding));
        }
        if round>=plan.rounds(){return Err(failure(Cause::LogicalRound,retained,funding));}
        let (destination,origin)=plan.peers();
        let destination=i32::try_from(destination).map_err(|_|failure(Cause::LogicalRound,retained,funding))?;
        let origin=i32::try_from(origin).map_err(|_|failure(Cause::LogicalRound,retained,funding))?;
        let send_persistent=selection.persistent(source)?;
        let receive_persistent=selection.persistent(source)?;
        for persistent in [&send_persistent,&receive_persistent] {
            if persistent.native().has_unqualified_storage() || !persistent.native().is_for(group.native_group())
                || !persistent.source().same_source(retained) {
                return Err(failure(Cause::Resource,retained,funding));
            }
        }
        let send=group.native_group().cpu_layout_storage(shape,dtype,GroupWorkerOperation::Send{peer:destination})
            .map_err(|_|failure(Cause::Resource,retained,funding))?;
        let receive=group.native_group().cpu_layout_storage(receive_shape,dtype,GroupWorkerOperation::Receive{peer:origin})
            .map_err(|_|failure(Cause::Resource,retained,funding))?;
        funding.reserve_metadata(send.exchange_layout_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let native=send.with_asymmetric_exchange_layout(receive).map_err(|_|failure(Cause::Resource,retained,funding))?;
        Ok(Self{native,_persistent:[send_persistent,receive_persistent],selection,round,
            submission_order,source:retained.clone(),funding:funding.clone()})
    }
}

impl<'a> OriginalRouteLayoutRound<'a> {
    pub(crate) fn graph_capacity(&self)->usize { self.native.graph_capacity() }
    pub(crate) fn record_capacity(&self)->usize { self.native.record_capacity() }
    /// Complete paired physical requirement before an input or primitive exists.
    /// The actual accepted pair still recomputes backing from its settled input.
    pub(crate) fn backing_capacity(&self,source:&OriginalCommunicationSource<'_>,runtime:&PreparedInputRuntime)
        ->Result<usize,Error> {
        let controls=[size_of::<Self>(),size_of::<usize>(),size_of::<Result<usize,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&PreparedInputRuntime)>(),size_of::<(usize,usize)>(),
            size_of::<Option<usize>>(),
            self.native.send().backing_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            self.native.receive().backing_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let send=self.native.send().backing_capacity(runtime).map_err(|cause|failure(Cause::Buffer(cause),&self.source,&self.funding))?;
        let receive=self.native.receive().backing_capacity(runtime).map_err(|cause|failure(Cause::Buffer(cause),&self.source,&self.funding))?;
        send.checked_add(receive).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))
    }
    /// Bind the actual source through the shared native recomputation. Source
    /// pins precede H, and any failed prefix retains the original failure owner.
    pub(crate) fn bind_actual<'input>(self,source:&OriginalCommunicationSource<'_>,input:&'input Array)
        ->Result<OriginalRouteRound<'input>,Error> where 'a:'input {
        let parts=[size_of::<Self>(),size_of::<OriginalRouteRound<'input>>(),
            size_of::<Result<OriginalRouteRound<'input>,Error>>(),size_of::<(&OriginalCommunicationSource<'_>,&Array)>(),
            self.native.binding_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.source.same_source(source.source()) {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        let native=self.native.bind_actual(input).map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalRouteRound{native,_persistent:RoundPersistent::Queried(self._persistent),selection:self.selection,round:self.round,
            submission_order:self.submission_order,source:self.source,funding:self.funding})
    }
}

/// The exact queried pair plus the immutable table which paid and qualified
/// both persistent native sources. No Array or active role can enter this owner.
pub(crate) struct OwnedOriginalExchangeLayoutRound {
    native: safemlx::distributed::OwnedGroupCpuExchangeLayoutStorage,
    selection: ExchangeSelection,
    round: usize,
    submission_order: SubmissionOrder,
    backing: usize,
    owner: super::super::parallel::OriginalParallelSource,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
impl OriginalRouteLayoutRound<'_> {
    pub(crate) fn try_into_owned(self,owner:&super::super::parallel::OriginalParallelSource)->Result<OwnedOriginalExchangeLayoutRound,Error> {
        let controls=[size_of::<Self>(),size_of::<OwnedOriginalExchangeLayoutRound>(),
            size_of::<Result<OwnedOriginalExchangeLayoutRound,Error>>(),
            size_of::<super::super::parallel::OriginalParallelSource>(),
            self.native.ownership_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        if !owner.declaration_source().same_source(&self.source) || !owner.funding().same_account(&self.funding) {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        let backing={
            let source=owner.communication_source()?;
            let runtime=owner.agreement_inputs().ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?.runtime();
            self.backing_capacity(&source,runtime)?
        };
        let native=self.native.try_into_owned().map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OwnedOriginalExchangeLayoutRound{native,selection:self.selection,round:self.round,
            submission_order:self.submission_order,backing,owner:owner.clone(),source:self.source,funding:self.funding})
    }
}
impl OwnedOriginalExchangeLayoutRound {
    pub(crate) fn ordinary_controls(&self)->Option<safemlx::distributed::OrdinaryGroupControls>{
        self.native.ordinary_controls()
    }
    pub(crate) fn graph_capacity(&self)->usize{self.native.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.native.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize{self.backing}
    pub(crate) fn shape(&self)->&[i32]{self.native.shape()}
    pub(crate) fn receive_shape(&self)->&[i32]{self.native.receive_shape()}
    pub(crate) fn maximum_backing_births(&self)->Option<usize>{self.native.maximum_backing_births()}
    pub(crate) fn bind_actual<'a>(&'a self,source:&OriginalCommunicationSource<'_>,input:&'a Array)
        ->Result<OriginalRouteRound<'a>,Error>{
        self.bind_actual_pair(source, input, input)
    }
    pub(crate) fn bind_actual_pair<'a>(&'a self, source: &OriginalCommunicationSource<'_>,
        input: &'a Array, receive_like: &'a Array) -> Result<OriginalRouteRound<'a>, Error> {
        let controls=[size_of::<Self>(),size_of::<OriginalRouteRound<'a>>(),
            size_of::<(&Self, &OriginalCommunicationSource<'_>, &Array, &Array)>(),
            size_of::<Result<OriginalRouteRound<'a>,Error>>(),
            self.native.binding_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        let group=self.selection.group(source)?;
        if !source.source().same_source(&self.source)
            || !self.native.is_for_group(group.native_group()) {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        let native=self.native.bind_actual_pair(input,receive_like).map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalRouteRound{native,_persistent:RoundPersistent::Retained(&self.owner),selection:self.selection,
            round:self.round,submission_order:self.submission_order,source:self.source.clone(),funding:self.funding.clone()})
    }
}

/// Compatibility name for the same pair, whose private selection remains a route.
pub(crate) type OwnedOriginalRouteLayoutRound=OwnedOriginalExchangeLayoutRound;
