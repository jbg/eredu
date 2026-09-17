//! Explicit finite invocation of the existing demand and selected-window workers.
use super::*;
use crate::backend::runtime::{execution::generic::{OriginalSelectedResidencyAccess,
    OriginalSelectedResidencyAttempt},residency::manager::{SupplementaryResidencySource,
    ForegroundDiskWindowPlan,ForegroundDiskSubsetCeiling,ForegroundDiskSourceCapacity,ForegroundDiskSourceSeries,
    OriginalResidencySource,WindowPopulation}};
use eredu_runtime::{residency::ResidencyClosureSlot,working_memory::{WorkingMemoryPool,
    OriginalHostSourceBank,HostSourceConstructionFacts}};
use std::cell::RefCell;
use crate::backend::submission_recovery::{Retention,Status,observed::{PreparedObservedRecovery,ObservedRecovery}};
type PendingInvocation=PreparedObservedRecovery<InvocationRetention,WorkspaceMetadataFunding>;
type ActiveInvocation=ObservedRecovery<InvocationRetention,WorkspaceMetadataFunding>;

/// One actual unit invocation. Completed demands still select each exact window.
pub(crate) struct IndexedResidencyPlan {
    binding:IndexedBankSource,
    bank:SharedAddressableParameterBank,
    first:AddressableChunkCensus,
    revision:u64,
    native_window:WindowPopulation,
    source:SupplementaryResidencySource,
    manager:ResidencyManager,
    constructor_bytes:u64,
    read:Option<ForegroundDiskSubsetCeiling>,
    pool:Option<WorkingMemoryPool>,
    source_funding:WorkspaceMetadataFunding,
    funding:WorkspaceMetadataFunding,
}
/// Exactly one direct child of the accepted root per retained chunk.
/// Fields are private: scalar counts cannot manufacture these authorities.
pub(crate) struct IndexedConstructorPartitions {
    banks:Vec<OriginalHostSourceBank>,
}
impl IndexedResidencyPlan {
    pub(crate) fn with_native_copy_source<T>(&self,
        visit: impl FnOnce(&SupplementaryResidencySource, WindowPopulation, usize) -> T) -> T {
        visit(&self.source, self.native_window, self.first.plan().len())
    }
    fn clone_frame_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Result<Self,Error>>(),
            size_of::<(&Self,&WorkspaceMetadataFunding)>(),size_of::<IndexedBankSource>(),
            size_of::<SharedAddressableParameterBank>(),size_of::<SupplementaryResidencySource>(),
            size_of::<ResidencyManager>(),size_of::<Option<ForegroundDiskSubsetCeiling>>(),
            size_of::<Option<WorkingMemoryPool>>(),size_of::<WorkspaceMetadataFunding>()*2];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    /// All actual descriptive clone, parent/child source loan and factory
    /// wrapper controls. Constructor directories/read capacities are separate.
    pub(crate) fn invocation_control_bytes(&self)->Result<usize,Error> {
        let stream=StreamCopyPlan::<CopyCustody>::capture_control_bytes()
            .map_err(|_|self.failure(Cause::Identity))?;
        let parts=[Self::clone_frame_bytes(),
            if self.read.is_some(){ForegroundDiskSubsetCeiling::clone_control_bytes()}else{Some(0)},
            OriginalSelectedResidencyAccess::native_role_control_bytes(),
            crate::backend::runtime::execution::generic::PreparedSelectedResidencyAccess::bind_control_bytes(),
            Self::factory_control_bytes(stream),
            OriginalIndexedResidencyInvocation::preparation_control_bytes(stream),
            Some(OriginalIndexedResidencyFactory::request_control_bytes())];
        parts.into_iter().try_fold(size_of_val(&parts),|sum,n|sum.checked_add(n?))
            .ok_or_else(||self.failure(Cause::Overflow))
    }
    fn factory_control_bytes(stream_controls:usize)->Option<usize> {
        OriginalIndexedResidencyFactory::control_bytes()?.checked_add(stream_controls)
    }
    /// Reuses only the immutable source description. Accepted constructor/read
    /// banks and native occurrence state remain separate move-only inputs.
    pub(crate) fn clone_for_invocation(&self,funding:&WorkspaceMetadataFunding)->Result<Self,Error> {
        let fail=|cause|failed(cause,&self.bank,funding,None);
        funding.reserve_metadata(Self::clone_frame_bytes().ok_or_else(||fail(Cause::Overflow))?)
            .map_err(|cause|fail(Cause::Funding(cause)))?;
        Ok(Self{binding:self.binding.clone(),bank:self.bank.clone(),first:self.first,revision:self.revision,
            source:self.source.clone(),manager:self.manager.clone(),native_window:self.native_window,constructor_bytes:self.constructor_bytes,
            read:self.read.as_ref().map(|read|read.clone_for_invocation(funding)).transpose()
                .map_err(|cause|fail(Cause::Funding(cause)))?,pool:self.pool.clone(),source_funding:self.source_funding.clone(),funding:funding.clone()})
    }
    fn constructor_partition_frames()->Option<usize> {
        let frames=[size_of::<IndexedConstructorPartitions>(),size_of::<Result<IndexedConstructorPartitions,Error>>(),
            size_of::<(&Self,&OriginalSelectedResidencyAccess,&mut OriginalHostSourceBank)>(),
            size_of::<[usize;3]>(),size_of::<u64>(),size_of::<HostSourceConstructionFacts>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    /// Actual direct-child directory and its shared construction transports.
    pub(crate) fn constructor_partition_control_bytes(&self)->Option<usize> {
        Self::constructor_partition_frames()?.checked_add(
            eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<OriginalHostSourceBank>(self.first.plan().len())?)
    }
    /// Direct root splitting. A region child is never treated as another root.
    pub(crate) fn partition_constructors(&self,access:&OriginalSelectedResidencyAccess,
        root:&mut OriginalHostSourceBank)->Result<IndexedConstructorPartitions,Error> {
        access.validate_source_bank(root).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        let count=self.first.plan().len();
        let attempts=count.checked_mul(OriginalSelectedResidencyAttempt::construction_attempts())
            .ok_or_else(||self.failure(Cause::Overflow))?;
        let bytes=self.constructor_bytes.checked_mul(u64::try_from(count).map_err(|_|self.failure(Cause::Overflow))?)
            .ok_or_else(||self.failure(Cause::Overflow))?;
        if root.remaining_bytes()<bytes||root.remaining_attempts()<attempts||root.remaining_partitions()<count {
            return Err(self.failure(Cause::Identity));
        }
        self.funding.reserve_metadata(Self::constructor_partition_frames().ok_or_else(||self.failure(Cause::Overflow))?)
            .map_err(|cause|self.failure(Cause::Funding(cause)))?;
        let mut banks=self.funding.metadata_vec(count).map_err(|cause|self.failure(Cause::Consumer(Error::Neural(cause))))?;
        for _ in 0..count {
            let bank=root.split(self.constructor_bytes,OriginalSelectedResidencyAttempt::construction_attempts())
                .map_err(|cause|self.failure(Cause::Budget(cause)))?;
            let facts=HostSourceConstructionFacts::new(self.constructor_bytes,
                OriginalSelectedResidencyAttempt::construction_attempts(),0).map_err(|_|self.failure(Cause::Geometry))?;
            access.validate_population(&bank,facts).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
            banks.push(bank);
        }
        Ok(IndexedConstructorPartitions{banks})
    }
    fn validate_partitions(&self,access:&OriginalSelectedResidencyAccess,parts:&IndexedConstructorPartitions)
        ->Result<(),Error> {
        if parts.banks.len()!=self.first.plan().len(){return Err(self.failure(Cause::Identity));}
        let facts=HostSourceConstructionFacts::new(self.constructor_bytes,
            OriginalSelectedResidencyAttempt::construction_attempts(),0).map_err(|_|self.failure(Cause::Geometry))?;
        for bank in &parts.banks {access.validate_population(bank,facts)
            .map_err(|cause|self.failure(Cause::Consumer(cause)))?;}
        Ok(())
    }
    pub(crate) fn constructor_facts(&self)->Option<HostSourceConstructionFacts> {
        let calls=self.first.plan().len();
        HostSourceConstructionFacts::new(self.constructor_bytes.checked_mul(u64::try_from(calls).ok()?)?,
            calls.checked_mul(OriginalSelectedResidencyAttempt::construction_attempts())?,calls).ok()
    }
    pub(crate) fn read_facts(&self)->Option<HostSourceConstructionFacts> {
        self.read.as_ref()?.source_facts(self.first.plan().len())
    }
    pub(crate) fn read_source_matches(&self,series:&ForegroundDiskSourceSeries)->bool {
        self.read.as_ref().is_some_and(|source|series.matches_ceiling(source))
    }
    /// Accumulates only this actual source's finite sequential region facts.
    pub(crate) fn include_read_source(&self,series:&mut Option<ForegroundDiskSourceSeries>)->Result<(),Error> {
        let Some(source)=&self.read else{return Ok(());};
        self.funding.reserve_metadata(ForegroundDiskSourceSeries::control_bytes().ok_or_else(||self.failure(Cause::Overflow))?)
            .map_err(|cause|self.failure(Cause::Funding(cause)))?;
        match series {
            Some(series)=>series.include(source,self.first.plan().len()).ok_or_else(||self.failure(Cause::Identity))?,
            None=>*series=Some(ForegroundDiskSourceSeries::new(source,self.first.plan().len())
                .ok_or_else(||self.failure(Cause::Overflow))?),
        }
        Ok(())
    }
    fn prepare_reads(&self,access:&OriginalSelectedResidencyAccess,reads:Option<OriginalHostSourceBank>)
        ->Result<Option<ForegroundDiskSourceCapacity>,Error> {
        match (&self.read,reads) {
            (Some(source),Some(bank))=>{
                if !source.matches_manager(&self.manager){return Err(self.failure(Cause::Identity));}
                access.prepare_read_capacity(source,self.first.plan().len(),bank).map(Some)
                    .map_err(|cause|self.failure(Cause::Consumer(cause)))
            },
            (None,None)=>Ok(None),_=>Err(self.failure(Cause::Identity)),
        }
    }
    fn validate_reads(&self,access:&OriginalSelectedResidencyAccess,reads:&Option<ForegroundDiskSourceCapacity>)->Result<(),Error> {
        match (&self.read,reads) {
            (Some(source),Some(capacity)) if source.matches_manager(&self.manager)=>
                access.validate_read_capacity(source,capacity).map_err(|cause|self.failure(Cause::Consumer(cause))),
            (None,None)=>Ok(()),_=>Err(self.failure(Cause::Identity)),
        }
    }
    pub(crate) fn requires_reads(&self)->bool {self.read.is_some()}
    pub(crate) fn census(&self)->AddressableChunkCensus {self.first}
    fn failure(&self,cause:Cause)->Error{failed(cause,&self.bank,&self.funding,None)}
    /// Banks come from the accepted role; metadata funding creates no authority.
    pub(crate) fn prepare(self,access:OriginalSelectedResidencyAccess,
        constructors:OriginalHostSourceBank,reads:Option<OriginalHostSourceBank>,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
    )->Result<OriginalIndexedResidencyInvocation,Error> {
        access.validate_population(&constructors,self.constructor_facts().ok_or_else(||self.failure(Cause::Overflow))?)
            .map_err(|e|self.failure(Cause::Consumer(e)))?;
        let mut constructors=constructors;
        let parts=self.partition_constructors(&access,&mut constructors)?;
        let capacity=self.prepare_reads(&access,reads)?;
        self.prepare_partitioned(access,parts,capacity,runtime,observer,stream)
    }
    fn prepare_partitioned(self,access:OriginalSelectedResidencyAccess,
        constructors:IndexedConstructorPartitions,capacity:Option<ForegroundDiskSourceCapacity>,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
    )->Result<OriginalIndexedResidencyInvocation,Error> {
        access.validate_observer(observer).map_err(|e|self.failure(Cause::Consumer(e)))?;
        self.validate_partitions(&access,&constructors)?;
        self.manager.validate_supplementary_source(&self.source).map_err(|e|self.failure(Cause::Source(e)))?;
        let selected=StreamCopyPlan::<CopyCustody>::capture(stream).map_err(|_|self.failure(Cause::Identity))?;
        let bytes=OriginalIndexedResidencyInvocation::preparation_control_bytes(selected.control_bytes().ok_or_else(||self.failure(Cause::Overflow))?)
            .ok_or_else(||self.failure(Cause::Overflow))?;
        self.funding.reserve_metadata(bytes).map_err(|e|self.failure(Cause::Funding(e)))?;
        self.validate_reads(&access,&capacity)?;
        let pending=PendingInvocation::new(self.funding.clone());
        Ok(OriginalIndexedResidencyInvocation(Rc::new(Invocation {
            pending:RefCell::new(Some(pending)),plan:self,access,runtime:runtime.inspection_alias(),observer:observer.clone(),stream:selected,
            state:RefCell::new(InvocationState{constructors:constructors.banks.into_iter(),next:0,failed:false,installed:false,closed:false}),capacity,
        })))
    }
}
impl IndexedBankSource {
    /// Work scales with the retained member declarations, never token rows or
    /// the combinatorial population of possible selected-demand subsets.
    pub(crate) fn inspect_original_residency(&self,first:AddressableChunkCensus,
        pool:Option<&WorkingMemoryPool>,funding:&WorkspaceMetadataFunding)->Result<IndexedResidencyPlan,Error> {
        let binding=self;
        let fail=|cause|failed(cause,&binding.bank,funding,None);
        let controls=[size_of::<IndexedResidencyPlan>(),size_of::<Result<IndexedResidencyPlan,Error>>(),
            size_of::<(AddressableChunkCensus,&Self,Option<&WorkingMemoryPool>,&WorkspaceMetadataFunding)>(),
            size_of::<(ResidencyManager,SupplementaryResidencySource,Vec<OffloadUnitId>,u64)>(),
            size_of::<Vec<ResidencyClosureSlot>>(),size_of::<[WindowPopulation;2]>(),
            size_of::<Option<ForegroundDiskWindowPlan>>(),size_of::<Option<ForegroundDiskSubsetCeiling>>(),
            size_of::<[usize;8]>(),size_of::<[u64;3]>(),size_of::<[u8;ParameterBankKey::unit_id_buffer_bytes()]>(),
            AddressableChunkPlan::control_bytes(),OriginalIndexedResidencyInvocation::control_bytes()
                .ok_or_else(||fail(Cause::Overflow))?,
            eredu_nn::Error::retained_source_control_bytes::<Failure>().ok_or_else(||fail(Cause::Overflow))?];
        funding.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls),usize::checked_add)
            .ok_or_else(||fail(Cause::Overflow))?).map_err(|e|fail(Cause::Funding(e)))?;
        if first.index()!=0{return Err(fail(Cause::Identity));}
        let (manager,source,eligible,revision)=binding.bank.with_workspace_source(funding,|bank| {
            let (members,maximum)=bank.unit_population(first.bank(),first.unit()).ok_or_else(||fail(Cause::Geometry))?;
            let expected=AddressableChunkPlan::new(first.total_rows(),first.routes(),members,first.access(),
                Some(maximum),binding.options.prefill_compact_bank_target_bytes()).map_err(|_|fail(Cause::Geometry))?;
            if expected!=first.plan() || !bank.same_source(&binding.bank){return Err(fail(Cause::Identity));}
            let manager=bank.manager();
            let source=manager.supplementary_residency_source().ok_or_else(||fail(Cause::Identity))?;
            let mut eligible=funding.metadata_vec(members).map_err(|e|fail(Cause::Consumer(Error::Neural(e))))?;
            let bytes=bank.unit_members(first.bank(),first.unit()).try_fold(0usize,|n,(key,_)|n.checked_add(key.unit_id_length()))
                .ok_or_else(||fail(Cause::Overflow))?;
            funding.reserve_metadata(bytes).map_err(|e|fail(Cause::Funding(e)))?;
            for (key,_) in bank.unit_members(first.bank(),first.unit()){eligible.push(key.unit_id());}
            if eligible.len()!=members{return Err(fail(Cause::Identity));}
            Ok::<_,Error>((manager.clone(),source.clone(),eligible,bank.parameter_revision()))
        }).map_err(|e|fail(Cause::Bank(e)))??;
        let units=source.source().controller_units;
        let source_bytes=OriginalResidencySource::constructor_storage_bytes(units,eligible.len().checked_add(1)
            .ok_or_else(||fail(Cause::Overflow))?).and_then(|n|usize::try_from(n).ok())
            .and_then(|n|n.checked_add(source.projection_control_bytes()?)).ok_or_else(||fail(Cause::Overflow))?;
        funding.reserve_metadata(source_bytes).map_err(|e|fail(Cause::Funding(e)))?;
        let mut scratch=funding.metadata_vec(units).map_err(|e|fail(Cause::Consumer(Error::Neural(e))))?;
        scratch.resize_with(units,ResidencyClosureSlot::default);
        let population=manager.selected_operation_ceiling(&source,&eligible,first.maximum_members(),&mut scratch)
            .map_err(|e|fail(Cause::Source(e)))?;
        let mut constructor_bytes=OriginalSelectedResidencyAttempt::construction_bytes(&source,population)
            .ok_or_else(||fail(Cause::Overflow))?;
        let read=if source.foreground().is_some() {
            let pool=pool.ok_or_else(||fail(Cause::Identity))?;
            let full=manager.selected_operation_population(&source,&eligible,&mut scratch).map_err(|e|fail(Cause::Source(e)))?;
            let disk=ForegroundDiskWindowPlan::new_with_metadata(&manager,pool,full,&eligible,&mut scratch,Some(funding))?;
            funding.reserve_metadata(ForegroundDiskSubsetCeiling::control_bytes().ok_or_else(||fail(Cause::Overflow))?)
                .map_err(|e|fail(Cause::Funding(e)))?;
            let ceiling=disk.subset_ceiling(population.units).ok_or_else(||fail(Cause::Overflow))?;
            constructor_bytes=constructor_bytes.checked_add(ceiling.attempt_control_bytes()).ok_or_else(||fail(Cause::Overflow))?;
            Some(ceiling)
        }else{
            manager.validate_supplementary_host(&source).map_err(|e|fail(Cause::Host(e)))?;
            None
        };
        Ok(IndexedResidencyPlan{binding:binding.clone(),bank:binding.bank.clone(),first,revision,native_window:population,source,manager,constructor_bytes,
            pool:read.as_ref().and(pool.cloned()),read,source_funding:funding.clone(),funding:funding.clone()})
    }
}
struct InvocationState {
    constructors:std::vec::IntoIter<OriginalHostSourceBank>,
    next:usize,
    failed:bool,
    installed:bool,
    closed:bool,
}
struct Invocation {
    plan:IndexedResidencyPlan,
    access:OriginalSelectedResidencyAccess,
    runtime:PreparedInputRuntime,
    observer:OriginalScopeObserver,
    stream:StreamCopyPlan<CopyCustody>,
    state:RefCell<InvocationState>,
    capacity:Option<ForegroundDiskSourceCapacity>,
    pending:RefCell<Option<PendingInvocation>>,
}
#[derive(Clone)]
pub(crate) struct OriginalIndexedResidencyInvocation(Rc<Invocation>);
// This payload is thread-local to the existing recovery queue. It never
// enters a Send/Sync public error and owns no independent native authority.
struct InvocationRetention {
    child:Option<OriginalIndexedChunkSource>,
    invocation:OriginalIndexedResidencyInvocation,
}
impl Retention for InvocationRetention {fn observe(&self,_:Status){}}
impl OriginalIndexedResidencyInvocation {
    fn preparation_control_bytes(stream_controls:usize)->Option<usize> {
        Self::control_bytes()?.checked_add(stream_controls)
    }
    fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Invocation>(),size_of::<InvocationState>(),
            size_of::<InvocationRetention>(),size_of::<Option<OriginalIndexedChunkSource>>(),
            size_of::<Option<Self>>(),size_of::<Result<Self,Error>>(),
            size_of::<std::cell::RefMut<'_,InvocationState>>(),size_of::<Result<(),Error>>(),
            size_of::<Result<OriginalHostSourceBank,eredu_runtime::working_memory::HostDestinationCause>>(),
            size_of::<Result<OriginalIndexedChunkSource,Error>>(),OriginalSelectedResidencyAccess::control_bytes()?,
            Layout::new::<[usize;2]>().extend(Layout::new::<Invocation>()).ok()?.0.pad_to_align().size(),
            PreparedInputRuntime::inspection_alias_control_bytes(),OriginalScopeObserver::control_bytes()?,
            usize::try_from(PendingInvocation::control_bytes::<Error>()?).ok()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    fn fail(&self,cause:Cause)->Error{self.0.plan.failure(cause)}
    fn validate(&self,movement:&MlxIndexedMovement,stream:&Stream)->Result<(),Error> {
        self.0.access.validate_observer(&self.0.observer)?;
        if !self.0.stream.matches_source(stream){return Err(self.fail(Cause::Identity));}
        let binding=movement.binding.as_ref().ok_or_else(||self.fail(Cause::Identity))?;
        if !binding.same_binding(&self.0.plan.binding){return Err(self.fail(Cause::Identity));}
        binding.bank.with_workspace_source(&self.0.plan.funding,|source| {
            if !source.same_source(&self.0.plan.bank)||source.parameter_revision()!=self.0.plan.revision {
                return Err(self.fail(Cause::Identity));
            }
            Ok(())
        }).map_err(|e|self.fail(Cause::Bank(e)))?
    }
    /// Funds the shared parent adapter under the explicitly installed source.
    /// Native graph/backing capacity remains the enclosing parent recipe's.
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn reserve_parent_adapter(
        &self,movement:&MlxIndexedMovement,stream:&Stream,adapter:Option<usize>,arguments:usize,
    )->Result<(),Error> {
        self.validate(movement,stream)?;
        let state=self.0.state.try_borrow().map_err(|_|self.fail(Cause::Spent))?;
        if !state.installed||state.closed||state.failed{return Err(self.fail(Cause::Spent));}
        let frames=[size_of::<(&Self,&MlxIndexedMovement,&Stream,Option<usize>,usize)>(),
            size_of::<std::cell::Ref<'_,InvocationState>>(),size_of::<Result<(),Error>>(),
            size_of::<Result<MlxTensor,eredu_nn::Error>>(),size_of::<Result<MlxTensor,Error>>(),
            adapter.ok_or_else(||self.fail(Cause::Overflow))?,arguments];
        let bytes=frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.fail(Cause::Overflow))?;
        self.0.plan.funding.reserve_metadata(bytes).map_err(|cause|self.fail(Cause::Funding(cause)))
    }
    /// The ordinal is spent before construction; an owning failure cannot retry.
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn begin_chunk(&self,movement:&mut MlxIndexedMovement,indices:&MlxTensor,
        census:AddressableChunkCensus,stream:&Stream)->Result<(),Error> {
        self.validate(movement,stream)?;
        let p=&self.0.plan;
        let mut state=self.0.state.try_borrow_mut().map_err(|_|self.fail(Cause::Spent))?;
        if state.failed||state.closed||!state.installed {return Err(self.fail(Cause::Spent));}
        let expected=AddressableChunkCensus::new(p.first.bank(),p.first.unit(),p.first.plan(),state.next,p.first.access());
        if expected!=Some(census){state.failed=true;return Err(self.fail(Cause::Identity));}
        if let Some(previous)=movement.original.as_ref() {
            if !previous.body().completed.get(){state.failed=true;return Err(self.fail(Cause::Spent));}
        }
        if let Some(previous)=movement.original.take(){previous.body().closed.set(true);drop(previous);}
        state.next=state.next.checked_add(1).ok_or_else(||self.fail(Cause::Overflow))?;
        let child_bank=match state.constructors.next() {
            Some(bank)=>bank,None=>{state.failed=true;return Err(self.fail(Cause::Spent));}
        };
        drop(state);
        let result=(|| {
            let source=movement.prepare_original_chunk(census,indices,&self.0.runtime,&self.0.observer,stream,&p.funding)?;
            source.bind_selected_residency_source(self.0.access.clone(),child_bank,
                p.pool.as_ref().zip(self.0.capacity.as_ref()).map(|(pool,capacity)|(pool.clone(),capacity.clone())),stream)?;
            Ok(source)
        })();
        match result {
            Ok(source)=>{movement.original=Some(source);Ok(())},
            Err(cause)=>{self.0.state.borrow_mut().failed=true;Err(cause)},
        }
    }
}
struct InvocationGuard<'a,P> {
    owner:&'a mut P,
    movement:fn(&mut P)->&mut MlxIndexedMovement,
    invocation:OriginalIndexedResidencyInvocation,
    recovery:Option<ActiveInvocation>,
}
impl<P> Drop for InvocationGuard<'_,P> {
    fn drop(&mut self) {
        let movement=(self.movement)(self.owner);
        let child=movement.original.take();
        if let Some(child)=&child{child.body().closed.set(true);}
        if let Some(recovery)=self.recovery.as_mut(){recovery.retention_mut().child=child;}
        movement.invocation.take();
        if let Ok(mut state)=self.invocation.0.state.try_borrow_mut(){state.closed=true;}
        // Drop moves the already-prepared node to unchanged recovery. Native
        // completion, errors and callback panic cannot prematurely release it.
        drop(self.recovery.take());
    }
}
impl OriginalIndexedResidencyInvocation {
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn matches_funding(
        &self,funding:&WorkspaceMetadataFunding)->bool {self.0.plan.funding.same_account(funding)}
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn run_with_owner<P,R,E,F>(
        self,owner:&mut P,movement:fn(&mut P)->&mut MlxIndexedMovement,stream:&Stream,run:F)
        ->Result<Result<R,E>,Error> where F:FnOnce(&mut P)->Result<R,E> {
        Self::run_owner(owner,movement,self,stream,run)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn owner_control_bytes<P,R,E,F>()->Option<usize> {
        let frames=[size_of::<R>(),size_of::<E>(),size_of::<F>(),size_of::<Result<R,E>>(),
            size_of::<Result<Result<R,E>,Error>>(),size_of::<InvocationGuard<'_,P>>(),
            size_of::<(&mut P,fn(&mut P)->&mut MlxIndexedMovement)>(),
            size_of::<Option<PendingInvocation>>(),size_of::<std::cell::RefMut<'_,Option<PendingInvocation>>>(),
            Self::control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    fn run_owner<P,R,E,F>(owner:&mut P,movement:fn(&mut P)->&mut MlxIndexedMovement,
        invocation:Self,stream:&Stream,run:F)->Result<Result<R,E>,Error>
    where F:FnOnce(&mut P)->Result<R,E> {
        invocation.validate(movement(owner),stream)?;
        invocation.0.plan.funding.reserve_metadata(Self::owner_control_bytes::<P,R,E,F>()
            .ok_or_else(||invocation.fail(Cause::Overflow))?).map_err(|e|invocation.fail(Cause::Funding(e)))?;
        if movement(owner).original.is_some()||movement(owner).invocation.is_some(){return Err(invocation.fail(Cause::Spent));}
        {
            let mut state=invocation.0.state.try_borrow_mut().map_err(|_|invocation.fail(Cause::Spent))?;
            if state.installed||state.closed||state.failed{return Err(invocation.fail(Cause::Spent));}
            state.installed=true;
        }
        let pending=invocation.0.pending.try_borrow_mut().map_err(|_|invocation.fail(Cause::Spent))?
            .take().ok_or_else(||invocation.fail(Cause::Spent))?;
        let recovery=pending.activate(InvocationRetention{child:None,invocation:invocation.clone()},invocation.0.observer.clone());
        movement(owner).invocation=Some(invocation.clone());
        let guard=InvocationGuard{owner,movement,invocation:invocation.clone(),recovery:Some(recovery)};
        let result=run(&mut *guard.owner);
        let complete=invocation.0.state.try_borrow().is_ok_and(|state|
            !state.failed && state.next==invocation.0.plan.first.plan().len())
            && (guard.movement)(guard.owner).original.as_ref().is_some_and(|source|source.body().completed.get());
        drop(guard);
        match result {
            Ok(value) if complete=>Ok(Ok(value)),
            Ok(_)=>Err(invocation.fail(Cause::Identity)),
            Err(cause)=>Ok(Err(cause)),
        }
    }
}
impl MlxIndexedMovement {
    /// Native compatibility entry for a direct lexical movement callback.
    pub(crate) fn with_original_residency_invocation<T,F>(&mut self,
        invocation:OriginalIndexedResidencyInvocation,stream:&Stream,run:F)->Result<T,Error>
    where F:FnOnce(&mut Self)->Result<T,Error> {
        OriginalIndexedResidencyInvocation::run_owner(self,|value|value,invocation,stream,run)?
    }
}

impl MlxIndexedMovement {
    /// Compatibility projection from this actual selected movement binding.
    pub(crate) fn inspect_original_residency(&self,first:AddressableChunkCensus,
        pool:Option<&WorkingMemoryPool>,funding:&WorkspaceMetadataFunding)->Result<IndexedResidencyPlan,Error> {
        let binding=self.binding.as_ref().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?;
        if self.original.is_some()||self.invocation.is_some(){return Err(failed(Cause::Spent,&binding.bank,funding,None));}
        binding.inspect_original_residency(first,pool,funding)
    }
}

/// Move-only late constructor under an already accepted region. Its typed plan
/// came from model binding; only the actual movement can authenticate/consume it.
pub(crate) struct OriginalIndexedResidencyFactory {
    plan:IndexedResidencyPlan,
    access:OriginalSelectedResidencyAccess,
    constructors:IndexedConstructorPartitions,
    reads:Option<ForegroundDiskSourceCapacity>,
    runtime:PreparedInputRuntime,
    observer:OriginalScopeObserver,
    stream:StreamCopyPlan<CopyCustody>,
}
impl IndexedResidencyPlan {
    /// The outer numerical source prepares this factory without inspecting a
    /// generic architecture owner or taking its mutable movement field.
    pub(crate) fn prepare_factory(self,access:OriginalSelectedResidencyAccess,
        constructors:OriginalHostSourceBank,reads:Option<OriginalHostSourceBank>,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
    )->Result<OriginalIndexedResidencyFactory,Error> {
        access.validate_population(&constructors,self.constructor_facts().ok_or_else(||self.failure(Cause::Overflow))?)
            .map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        let mut constructors=constructors;
        let parts=self.partition_constructors(&access,&mut constructors)?;
        self.prepare_factory_partitioned(access,parts,reads,runtime,observer,stream)
    }
    /// Consumes children previously issued directly by the accepted model root.
    pub(crate) fn prepare_factory_partitioned(self,access:OriginalSelectedResidencyAccess,
        constructors:IndexedConstructorPartitions,reads:Option<OriginalHostSourceBank>,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
    )->Result<OriginalIndexedResidencyFactory,Error> {
        let capacity=self.prepare_reads(&access,reads)?;
        self.prepare_factory_shared(access,constructors,capacity,runtime,observer,stream)
    }
    /// Shares the one accepted actual reader capacity across sequential regions.
    pub(crate) fn prepare_factory_shared(self,access:OriginalSelectedResidencyAccess,
        constructors:IndexedConstructorPartitions,reads:Option<ForegroundDiskSourceCapacity>,
        runtime:&PreparedInputRuntime,observer:&OriginalScopeObserver,stream:&Stream,
    )->Result<OriginalIndexedResidencyFactory,Error> {
        let source=StreamCopyPlan::<CopyCustody>::capture(stream).map_err(|_|self.failure(Cause::Identity))?;
        let bytes=source.control_bytes().and_then(Self::factory_control_bytes)
            .ok_or_else(||self.failure(Cause::Overflow))?;
        self.funding.reserve_metadata(bytes).map_err(|e|self.failure(Cause::Funding(e)))?;
        access.validate_observer(observer).map_err(|e|self.failure(Cause::Consumer(e)))?;
        self.validate_partitions(&access,&constructors)?;
        self.validate_reads(&access,&reads)?;
        Ok(OriginalIndexedResidencyFactory{plan:self,access,constructors,reads,
            runtime:runtime.inspection_alias(),observer:observer.clone(),stream:source})
    }
}
impl OriginalIndexedResidencyFactory {
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn funding(
        &self)->&WorkspaceMetadataFunding { &self.plan.funding }
    /// Checks the reached immutable coordinates before consuming the native
    /// factory. Numerical/source-layout identity remains the outer quote's job.
    fn request_control_bytes()->usize {
        size_of::<(
            &Self, &eredu_runtime::expert::IndexedInvocationRequest<'_,MlxTensor>,
            eredu_nn::workspace::WorkspaceAddressableRegionView<'_>,
            eredu_nn::workspace::ExpertRegionInputShape,
            Result<(),Error>,
        )>()
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn validate_request(
        &self, request:&eredu_runtime::expert::IndexedInvocationRequest<'_,MlxTensor>,
    )->Result<(),Error> {
        let declaration=request.declaration;
        let first=self.plan.first;
        self.plan.funding.reserve_metadata(Self::request_control_bytes())
            .map_err(|cause|self.plan.failure(Cause::Funding(cause)))?;
        declaration.validate().map_err(|_|self.plan.failure(Cause::Geometry))?;
        let shape=eredu_nn::workspace::ExpertRegionInputShape::inspect(
            request.input.shape(),request.routes.group_indices().shape())
            .map_err(|_|self.plan.failure(Cause::Geometry))?;
        if declaration.bank as usize!=first.bank() || declaration.unit!=first.unit()
            || declaration.chunks!=first.plan().workspace_source()
            || declaration.prefill!=(first.access()==eredu_runtime::ParameterBankAccess::Bulk)
            || usize::try_from(shape.rows).ok()!=Some(first.total_rows())
            || usize::try_from(shape.routes).ok()!=Some(first.routes())
            || shape.width!=declaration.kernel.dimensions().0
            || request.routes.selected_scores().shape()!=request.routes.group_indices().shape()
            || request.routes.coefficients().shape()!=request.routes.group_indices().shape() {
            return Err(self.plan.failure(Cause::Identity));
        }
        Ok(())
    }

    pub(crate) fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<Option<Self>>(),size_of::<Result<Self,Error>>(),
            size_of::<Result<OriginalIndexedResidencyInvocation,Error>>(),size_of::<AddressableChunkPlan>(),
            size_of::<Result<AddressableChunkPlan,eredu_runtime::expert::AddressableChunkPlanError>>(),
            size_of::<(&MlxIndexedMovement,&Stream)>(),size_of::<[usize;4]>(),size_of::<u64>(),
            OriginalSelectedResidencyAccess::control_bytes()?,PreparedInputRuntime::inspection_alias_control_bytes(),
            OriginalScopeObserver::control_bytes()?];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn matches_funding(
        &self,funding:&WorkspaceMetadataFunding)->bool {self.plan.funding.same_account(funding)}
    pub(in crate::backend::runtime::residency::parameter_bank::movement) fn prepare(
        self,movement:&MlxIndexedMovement,stream:&Stream)->Result<OriginalIndexedResidencyInvocation,Error> {
        let binding=movement.binding.as_ref().ok_or_else(||self.plan.failure(Cause::Identity))?;
        if movement.original.is_some()||movement.invocation.is_some()||!self.stream.matches_source(stream)
            || !binding.same_binding(&self.plan.binding) {
            return Err(self.plan.failure(Cause::Identity));
        }
        binding.bank.with_workspace_source(&self.plan.funding,|source| {
            let first=self.plan.first;
            let (members,maximum)=source.unit_population(first.bank(),first.unit())
                .ok_or_else(||self.plan.failure(Cause::Geometry))?;
            let plan=AddressableChunkPlan::new(first.total_rows(),first.routes(),members,first.access(),Some(maximum),
                binding.options.prefill_compact_bank_target_bytes()).map_err(|_|self.plan.failure(Cause::Geometry))?;
            if !source.same_source(&self.plan.bank)||source.parameter_revision()!=self.plan.revision||plan!=first.plan() {
                return Err(self.plan.failure(Cause::Identity));
            }
            Ok::<_,Error>(())
        }).map_err(|e|self.plan.failure(Cause::Bank(e)))??;
        self.plan.prepare_partitioned(self.access,self.constructors,self.reads,&self.runtime,&self.observer,stream)
    }
}
