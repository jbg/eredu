//! One retained region occurrence; local numerical children borrow its source.
use super::*;
use crate::backend::nn::workspace::{ExpertLocalQuote, ExpertProviderWaveQuote, ExpertInactiveWaveQuote, ResidentExecutionMechanisms};
use eredu_nn::workspace::WorkspaceExpertRegionView;
enum ExpertOccurrenceQuote { Local(ExpertLocalQuote),ProviderWave(ExpertProviderWaveQuote),InactiveWave(ExpertInactiveWaveQuote) }
#[derive(Clone)]
pub(crate) struct RetainedExpertRegion { value: Option<Rc<ExpertOccurrenceQuote>>, funding: HostMetadataFunding }
impl RetainedExpertRegion {
    pub(crate) fn local(&self)->Option<&ExpertLocalQuote>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::Local(value)=>Some(value),_=>None}
    }
    pub(crate) fn value(&self) -> &ExpertLocalQuote { self.local().expect("validated local expert region") }
    pub(crate) fn wave(&self)->Option<&ExpertProviderWaveQuote>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::ProviderWave(value)=>Some(value),_=>None}
    }
    pub(crate) fn inactive(&self)->Option<&ExpertInactiveWaveQuote>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::InactiveWave(value)=>Some(value),_=>None}
    }
    fn counts(&self)->Option<&crate::backend::nn::workspace::ExpertCountQuote>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::Local(value)=>value.counts.as_ref(),
            ExpertOccurrenceQuote::InactiveWave(value)=>Some(&value.counts),_=>None}
    }
    fn transport(&self)->Option<&crate::backend::nn::workspace::ExpertTransportQuote>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::Local(value)=>value.transport.as_ref(),
            ExpertOccurrenceQuote::InactiveWave(value)=>Some(&value.transport),_=>None}
    }
    pub(crate) fn aggregate(&self)->Option<&crate::backend::nn::workspace::ExpertRegionAggregate>{
        match self.value.as_deref()?{ExpertOccurrenceQuote::Local(value)=>value.aggregate.as_ref(),
            ExpertOccurrenceQuote::InactiveWave(value)=>Some(&value.aggregate),_=>None}
    }
    fn source(&self)->&OriginalParallelSource{
        match self.value.as_deref().expect("retained expert source"){
            ExpertOccurrenceQuote::Local(value)=>value.source(),ExpertOccurrenceQuote::ProviderWave(value)=>value.source(),
            ExpertOccurrenceQuote::InactiveWave(value)=>value.source(),
        }
    }
    fn provider(&self)->Option<&crate::backend::nn::workspace::ExpertProviderQuote>{
        match self.value.as_deref()?{
            ExpertOccurrenceQuote::Local(value)=>value.provider.as_ref(),ExpertOccurrenceQuote::ProviderWave(value)=>Some(&value.provider),
            ExpertOccurrenceQuote::InactiveWave(value)=>Some(&value.provider),
        }
    }
    fn transfers(&self)->usize{self.transport().map_or(0,|value|value.profiles().len())}
}
impl Drop for RetainedExpertRegion {
    fn drop(&mut self) { if let Some(value)=self.value.take() { drop(Rc::into_inner(value)); } }
}
struct RegionCall { state: Option<Rc<State>> }
impl Drop for RegionCall {
    fn drop(&mut self) { if let Some(state)=self.state.take() { state.expert_active.set(None); drop(Rc::into_inner(state)); } }
}
impl OriginalParallelSource {
    pub(super) fn prepare_expert_occurrences(&self, operations: &[WorkspaceOperation],
        mechanism: Option<ResidentExecutionMechanisms>, addressable: Option<&crate::backend::nn::workspace::AddressableSources>,
        ordinary: Option<&crate::backend::nn::workspace::OrdinaryAddressableSources>) -> Result<Vec<ExpertOccurrence>, Error> {
        let count=operations.iter().filter(|op|matches!(op.kind,WorkspaceOperationKind::ExpertRegion(_)|WorkspaceOperationKind::ExpertProviderWave(_)|WorkspaceOperationKind::ExpertInactiveWave(_))).count();
        reserve(self.funding(), &[size_of::<Vec<ExpertOccurrence>>(),size_of::<Result<Vec<ExpertOccurrence>,Error>>(),
            Layout::array::<ExpertOccurrence>(count).map_err(|_|overflow())?.size(),failure_control_bytes().ok_or_else(overflow)?])?;
        let mut result=Vec::new();result.try_reserve_exact(count)
            .map_err(|_|failure(Cause::Resource,self.declaration_source(),self.funding()))?;
        for (ordinal, operation) in operations.iter().enumerate() {
            if !matches!(operation.kind,WorkspaceOperationKind::ExpertRegion(_)|WorkspaceOperationKind::ExpertProviderWave(_)|WorkspaceOperationKind::ExpertInactiveWave(_)){continue;}
            let mechanism=mechanism.ok_or_else(||failure(Cause::Resource,self.declaration_source(),self.funding()))?;
            let quote=match &operation.kind{
                WorkspaceOperationKind::ExpertRegion(_)=>ExpertOccurrenceQuote::Local(
                    if mechanism.uses_original_storage() {
                        ExpertLocalQuote::prepare(self,operation.as_view(),mechanism,addressable)
                    } else {
                        if addressable.is_some() { return Err(failure(Cause::Identity,self.declaration_source(),self.funding())); }
                        ExpertLocalQuote::prepare_ordinary(self,operation.as_view(),mechanism,ordinary)
                    }.map_err(Error::Neural)?),
                WorkspaceOperationKind::ExpertProviderWave(value)=>ExpertOccurrenceQuote::ProviderWave(
                    ExpertProviderWaveQuote::prepare(self,*value,mechanism).map_err(Error::Neural)?),
                WorkspaceOperationKind::ExpertInactiveWave(value)=>ExpertOccurrenceQuote::InactiveWave(
                    ExpertInactiveWaveQuote::prepare(self,**value,mechanism).map_err(Error::Neural)?),
                _=>return Err(failure(Cause::Identity,self.declaration_source(),self.funding())),
            };
            reserve(self.funding(), &[size_of::<RetainedExpertRegion>(),
                Layout::new::<[usize;2]>().extend(Layout::new::<ExpertOccurrenceQuote>()).map_err(|_|overflow())?.0.pad_to_align().size()])?;
            result.push(ExpertOccurrence{ordinal,quote:RetainedExpertRegion{value:Some(Rc::new(quote)),funding:self.funding().clone()}});
        }
        Ok(result)
    }
}
impl OriginalParallelBinding {
    pub(crate) fn with_expert_region<T,E,F>(&self,source:&OriginalParallelSource,
        declaration:WorkspaceExpertRegionView<'_>,stream:&Stream,run:F)->Result<Result<T,E>,Error>
    where F:FnOnce()->Result<Result<T,E>,Error> {
        self.with_expert_occurrence(source,stream,true,true,|quote|quote.local().is_some_and(|value|value.matches(declaration)),run)
    }
    pub(crate) fn with_expert_provider_wave<E,F>(&self,source:&OriginalParallelSource,
        declaration:eredu_nn::workspace::WorkspaceExpertProviderWave,stream:&Stream,run:F)
        ->Result<Result<(),E>,Error>
    where F:FnOnce()->Result<Result<(),E>,Error>{
        self.with_expert_occurrence(source,stream,false,false,|quote|quote.wave().is_some_and(|value|value.declaration==declaration),run)
    }
    pub(crate) fn with_expert_inactive_wave<E,F>(&self,source:&OriginalParallelSource,
        declaration:eredu_nn::workspace::WorkspaceExpertInactiveWave,stream:&Stream,run:F)
        ->Result<Result<(),E>,Error>
    where F:FnOnce()->Result<Result<(),E>,Error>{
        self.with_expert_occurrence(source,stream,false,true,|quote|quote.inactive().is_some_and(|value|value.declaration==declaration),run)
    }
    fn with_expert_occurrence<T,E,F,M>(&self,source:&OriginalParallelSource,stream:&Stream,local:bool,counts:bool,matches:M,run:F)
        ->Result<Result<T,E>,Error>
    where F:FnOnce()->Result<Result<T,E>,Error>,M:FnOnce(&RetainedExpertRegion)->bool{
        reserve(&self.funding,&[size_of::<M>(),size_of::<F>(),size_of::<T>(),size_of::<E>(),size_of::<RegionCall>(),
            size_of::<Result<Result<T,E>,Error>>(),size_of::<StreamCopyPlan<()>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let state=self.state.upgrade().ok_or_else(fail)?;
        let bound=state.bound.get().ok_or_else(fail)?;
        if state.closed.get()||state.forward_done.get()||state.calling.get()||state.expert_active.get().is_some()
            ||state.authority.ensure_active().is_err()||!state.quote_source.same_source(source)
            ||!bound.stream.matches_source(stream){return Err(fail());}
        let index=state.next_expert.get();let row=state.expert_regions.get(index).ok_or_else(fail)?;
        state.next_expert.set(index.checked_add(1).ok_or_else(overflow)?);
        if !matches(&row.quote)||!row.quote.source().same_source(source)
            ||state.occurrences.get(state.next.get()).is_some_and(|next|next.ordinal<row.ordinal)
            ||state.boundaries.get(state.next_boundary.get()).is_some_and(|next|next.ordinal<row.ordinal){return Err(fail());}
        state.expert_active.set(Some(index));
        state.expert_local_taken.set(false);
        state.expert_local_complete.set(!local);
        state.expert_transport_next.set(0);
        state.expert_transport_pending.set(false);
        state.expert_count_taken.set(false);
        state.expert_count_complete.set(!counts);
        state.expert_vote_next.set(0);
        state.expert_vote_pending.set(false);
        let call=RegionCall{state:Some(state)};
        let result=run();
        if matches!(&result,Ok(Ok(_))) {
            let state=call.state.as_ref().ok_or_else(fail)?;
            let expected=state.expert_regions.get(index).ok_or_else(fail)?.quote.transfers();
            if !state.expert_count_complete.get()||!state.expert_local_complete.get()||state.expert_transport_pending.get()
                ||state.expert_transport_next.get()!=expected||state.expert_vote_pending.get()
                ||state.expert_vote_next.get()!=row_votes(state,index).ok_or_else(fail)? {return Err(fail());}
        }
        result
    }
    pub(crate) fn expert_local_source(&self,source:&OriginalParallelSource,
        declaration:WorkspaceExpertRegionView<'_>,stream:&Stream)->Result<(RetainedExpertRegion,OriginalScopeObserver),Error>{
        reserve(&self.funding,&[size_of::<RetainedExpertRegion>(),size_of::<ControlSourceLoan>(),
            size_of::<Result<(RetainedExpertRegion,OriginalScopeObserver),Error>>(),OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let index=state.expert_active.get().ok_or_else(fail)?;
        let row=state.expert_regions.get(index).ok_or_else(fail)?;
        if state.expert_local_taken.replace(true) { return Err(fail()); }
        if state.closed.get()||state.authority.ensure_active().is_err()||!state.quote_source.same_source(source)
            ||!state.bound.get().is_some_and(|bound|bound.stream.matches_source(stream))
            ||!row.quote.local().is_some_and(|value|value.matches(declaration)){return Err(fail());}
        Ok((row.quote.clone(),state.bound.get().ok_or_else(fail)?.observer.clone()))
    }
    pub(crate) fn complete_expert_local(&self,source:&OriginalParallelSource,
        declaration:WorkspaceExpertRegionView<'_>,stream:&Stream)->Result<(),Error>{
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let row=state.expert_regions.get(state.expert_active.get().ok_or_else(fail)?).ok_or_else(fail)?;
        if state.closed.get()||!state.expert_local_taken.get()||state.expert_local_complete.get()
            ||state.authority.ensure_active().is_err()||!state.quote_source.same_source(source)
            ||!row.quote.local().is_some_and(|value|value.matches(declaration))
            ||!state.bound.get().is_some_and(|bound|bound.stream.matches_source(stream)){return Err(fail());}
        state.expert_local_complete.set(true);
        Ok(())
    }

}

impl OriginalParallelBinding {
    pub(crate) fn expert_movement_source(&self,source:&OriginalParallelSource,
        declaration:WorkspaceExpertRegionView<'_>,stream:&Stream)
        ->Result<(RetainedExpertRegion,OriginalScopeObserver,usize),Error> {
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<RetainedExpertRegion>(),
            size_of::<Result<(RetainedExpertRegion,OriginalScopeObserver,usize),Error>>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let index=state.expert_active.get().ok_or_else(fail)?;
        let row=state.expert_regions.get(index).ok_or_else(fail)?;
        if !row.quote.local().is_some_and(|value|value.matches(declaration)){return Err(fail());}
        self.validate_expert_movement(source,&row.quote,index,stream)?;
        Ok((row.quote.clone(),state.bound.get().ok_or_else(fail)?.observer.clone(),index))
    }
    pub(crate) fn validate_expert_movement(&self,source:&OriginalParallelSource,
        quote:&RetainedExpertRegion,index:usize,stream:&Stream)->Result<(),Error> {
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Result<(),Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let row=state.expert_regions.get(index).ok_or_else(fail)?;
        if state.closed.get()||state.forward_done.get()||state.expert_active.get()!=Some(index)
            ||state.authority.ensure_active().is_err()||!state.quote_source.same_source(source)
            ||!state.bound.get().is_some_and(|bound|bound.stream.matches_source(stream))
            ||!match(&row.quote.value,&quote.value){(Some(a),Some(b))=>Rc::ptr_eq(a,b),_=>false} {
            return Err(fail());
        }
        Ok(())
    }
}

/// A spent transfer occurrence carries the exact retained quote through native
/// construction/completion. It cannot authorize a later equal-shaped payload.
pub(crate) struct RetainedExpertTransfer {
    quote: RetainedExpertRegion,
    occurrence: usize,
    transfer: usize,
    local_next:Cell<usize>,
    pub(crate) profile: crate::backend::nn::workspace::ExpertTransferProfile,
}
impl RetainedExpertTransfer {
    pub(crate) fn requires_input_completion(&self)->bool{
        self.quote.inactive().is_some()||self.quote.transport()
            .is_some_and(|quote|quote.input_requires_parent_completion(self.transfer))
    }
    pub(crate) fn claim_local_stage(&self,kind:crate::backend::nn::workspace::ExpertLocalStage)
        ->Result<crate::backend::nn::workspace::ExpertLocalStageBound,Error>{
        let funding=&self.quote.funding;
        let source=self.quote.source();
        reserve(funding,&[size_of::<crate::backend::nn::workspace::ExpertLocalStageBound>(),
            size_of::<Result<crate::backend::nn::workspace::ExpertLocalStageBound,Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,source.declaration_source(),funding);
        let index=self.local_next.get();
        self.local_next.set(index.checked_add(1).ok_or_else(overflow)?);
        let stage=self.quote.transport().and_then(|quote|quote.local(self.transfer))
            .and_then(|quote|quote.stage(index)).ok_or_else(fail)?;
        if stage.kind!=kind{return Err(fail());}
        Ok(stage)
    }
}
impl OriginalParallelBinding {
    pub(crate) fn claim_expert_transfer(&self,source:&OriginalParallelSource,stream:&Stream)
        ->Result<Option<RetainedExpertTransfer>,Error> {
        reserve(&self.funding,&[size_of::<RetainedExpertTransfer>(),size_of::<ControlSourceLoan>(),
            size_of::<Result<Option<RetainedExpertTransfer>,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let Some(occurrence)=state.expert_active.get()else{return Ok(None);};
        let row=state.expert_regions.get(occurrence).ok_or_else(fail)?;
        self.validate_expert_movement(source,&row.quote,occurrence,stream)?;
        if state.expert_transport_pending.replace(true){return Err(fail());}
        let transfer=state.expert_transport_next.get();
        state.expert_transport_next.set(transfer.checked_add(1).ok_or_else(overflow)?);
        let profile=row.quote.transport().and_then(|quote|quote.profile(transfer)).ok_or_else(fail)?;
        if !state.expert_count_complete.get()||state.expert_vote_pending.get()
            ||(profile.payload.reverse()&&state.expert_vote_next.get()!=row.quote.provider().ok_or_else(fail)?.len())
            ||(!profile.payload.reverse()&&state.expert_vote_next.get()!=0){return Err(fail());}
        Ok(Some(RetainedExpertTransfer{quote:row.quote.clone(),occurrence,transfer,local_next:Cell::new(0),profile}))
    }
    pub(crate) fn complete_expert_transfer(&self,source:&OriginalParallelSource,
        transfer:&RetainedExpertTransfer,stream:&Stream)->Result<(),Error> {
        self.validate_expert_movement(source,&transfer.quote,transfer.occurrence,stream)?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Result<(),Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        if !state.expert_transport_pending.get()||state.expert_transport_next.get()!=transfer.transfer.checked_add(1).ok_or_else(overflow)? {
            return Err(fail());
        }
        if let Some(local)=transfer.quote.transport().and_then(|quote|quote.local(transfer.transfer)) {
            if transfer.local_next.get()!=local.len(){return Err(fail());}
        }
        state.expert_transport_pending.set(false);
        Ok(())
    }
}

/// One count-consensus attempt retains its selected region and all child recipes.
pub(crate) struct RetainedExpertCount { quote:RetainedExpertRegion, occurrence:usize }
impl RetainedExpertCount {
    pub(crate) fn matches(&self,order:usize,peers:usize,rank:usize)->bool {
        self.quote.counts().is_some_and(|count|count.matches(order,peers,rank))
    }
    pub(crate) fn logical(&self,step:Option<(usize,usize)>)->Option<super::RetainedLogicalCollective>{
        self.quote.counts()?.logical(step)
    }
}
impl OriginalParallelBinding {
    pub(crate) fn claim_expert_counts(&self,source:&OriginalParallelSource,group:&Group,stream:&Stream)
        ->Result<Option<RetainedExpertCount>,Error>{
        reserve(&self.funding,&[size_of::<RetainedExpertCount>(),size_of::<ControlSourceLoan>(),
            size_of::<Result<Option<RetainedExpertCount>,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let Some(occurrence)=state.expert_active.get()else{return Ok(None);};
        if state.expert_count_taken.replace(true){return Err(fail());}
        let row=state.expert_regions.get(occurrence).ok_or_else(fail)?;
        self.validate_expert_movement(source,&row.quote,occurrence,stream)?;
        let actual=source.communication_source()?;
        let order=actual.source().manifest().groups().iter().enumerate()
            .find_map(|(order,_)|actual.matches_group(order,group).then_some(order)).ok_or_else(fail)?;
        if !row.quote.counts().is_some_and(|count|count.matches(order,group.size(),group.rank())){
            return Err(fail());
        }
        Ok(Some(RetainedExpertCount{quote:row.quote.clone(),occurrence}))
    }
    pub(crate) fn complete_expert_counts(&self,source:&OriginalParallelSource,
        count:&RetainedExpertCount,world:bool,stream:&Stream)->Result<(),Error>{
        self.validate_expert_movement(source,&count.quote,count.occurrence,stream)?;
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        if !state.expert_count_taken.get()||state.expert_count_complete.get()
            ||!count.quote.counts().is_some_and(|count|count.has_world()==world){return Err(fail());}
        state.expert_count_complete.set(true);Ok(())
    }
}

fn row_votes(state:&State,index:usize)->Option<usize>{
    Some(state.expert_regions.get(index)?.quote.provider()?.len())
}
pub(crate) struct RetainedExpertVote { quote:RetainedExpertRegion, occurrence:usize, vote:usize }
impl RetainedExpertVote {
    pub(crate) fn logical(&self)->Option<super::RetainedLogicalCollective>{
        self.quote.provider()?.vote(self.vote)?.logical.clone()
    }
    pub(crate) fn selection(&self)->Option<(usize,usize,usize)>{
        let vote=self.quote.provider()?.vote(self.vote)?;
        Some((vote.order,vote.peers,vote.backing))
    }
}
impl OriginalParallelBinding {
    pub(crate) fn claim_expert_vote(&self,source:&OriginalParallelSource,group:&Group,stream:&Stream)
        ->Result<Option<RetainedExpertVote>,Error>{
        reserve(&self.funding,&[size_of::<RetainedExpertVote>(),size_of::<ControlSourceLoan>(),
            size_of::<Result<Option<RetainedExpertVote>,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        let Some(occurrence)=state.expert_active.get()else{return Ok(None);};
        if state.expert_vote_pending.replace(true){return Err(fail());}
        let vote=state.expert_vote_next.get();
        state.expert_vote_next.set(vote.checked_add(1).ok_or_else(overflow)?);
        let row=state.expert_regions.get(occurrence).ok_or_else(fail)?;
        self.validate_expert_movement(source,&row.quote,occurrence,stream)?;
        let expected=row.quote.provider().and_then(|provider|provider.vote(vote)).ok_or_else(fail)?;
        let forward=row.quote.transport().map_or(0,|transport|transport.profiles().iter()
            .take_while(|profile|!profile.payload.reverse()).count());
        if state.expert_transport_pending.get()||state.expert_transport_next.get()!=forward{return Err(fail());}
        let actual=source.communication_source()?;
        if !actual.matches_group(expected.order,group)||group.size()!=expected.peers
            ||actual.group(expected.order).is_none_or(|(_,descriptor,_)|descriptor.id()!=expected.group)
            ||!state.expert_count_complete.get(){return Err(fail());}
        Ok(Some(RetainedExpertVote{quote:row.quote.clone(),occurrence,vote}))
    }
    pub(crate) fn complete_expert_vote(&self,source:&OriginalParallelSource,vote:&RetainedExpertVote,stream:&Stream)
        ->Result<(),Error>{
        self.validate_expert_movement(source,&vote.quote,vote.occurrence,stream)?;
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let fail=||failure(Cause::Identity,&self.source,&self.funding);
        let loan=ControlSourceLoan(Some(self.state.upgrade().ok_or_else(fail)?));
        let state=loan.0.as_ref().ok_or_else(fail)?;
        if !state.expert_vote_pending.get()||state.expert_vote_next.get()!=vote.vote.checked_add(1).ok_or_else(overflow)?{
            return Err(fail());
        }
        state.expert_vote_pending.set(false);Ok(())
    }
}
