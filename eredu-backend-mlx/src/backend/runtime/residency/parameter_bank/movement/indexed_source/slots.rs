//! Explicit registered operation/attempt loan for the same addressable driver.
use super::*;
use crate::backend::runtime::execution::generic::{OriginalOperationAccess,OriginalResidencyAttempt,OriginalSelectedResidencyAttempt,OriginalSelectedResidencyAccess};
use eredu_runtime::expert::PreparedIndexedDemandLoan;
use eredu_nn::Tensor;

pub(super) trait IndexedResidency {
    fn acquire(&mut self,source:&OriginalIndexedChunkSource,demands:&IndexedDemandSource,
        entries:&[(ParameterBankKey,u64)],stream:&Stream)->Result<AcquiredParameterGroups,Error>;
}
struct RegisteredResidency<U:'static> {
    attempt:OriginalResidencyAttempt,
    access:OriginalOperationAccess<U>,
}
impl<U:'static> IndexedResidency for RegisteredResidency<U> {
    fn acquire(&mut self,source:&OriginalIndexedChunkSource,demands:&IndexedDemandSource,
        entries:&[(ParameterBankKey,u64)],stream:&Stream)->Result<AcquiredParameterGroups,Error> {
        self.access.with_residency(&mut self.attempt,|slots,observer| {
            if !observer.same_scope(&source.body().observer){return Err(source.failure(Cause::Identity));}
            source.acquire_from_slots(demands,entries,slots,stream)
        })
    }
}
impl OriginalIndexedChunkSource {
    /// Attaches the selected registered ordinal before discovery. Both owners
    /// are supplied by the request constructor; no cache slot is created here.
    pub(crate) fn bind_residency<U:'static>(&self,access:OriginalOperationAccess<U>,
        attempt:OriginalResidencyAttempt,stream:&Stream)->Result<(),Error> {
        self.validate_parent(stream)?;
        let b=self.body();
        access.validate_observer(&b.observer).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        let frames=[size_of::<RegisteredResidency<U>>(),size_of::<Box<RegisteredResidency<U>>>(),
            size_of::<Box<dyn IndexedResidency>>(),size_of::<Result<(),Error>>(),
            size_of::<Option<Box<dyn IndexedResidency>>>(),
            size_of::<std::cell::RefMut<'_,Option<Box<dyn IndexedResidency>>>>(),
            size_of::<OriginalOperationAccess<U>>(),size_of::<OriginalResidencyAttempt>(),
            Layout::new::<RegisteredResidency<U>>().size(),
            eredu_nn::Error::retained_source_control_bytes::<Failure>().ok_or_else(||self.failure(Cause::Overflow))?];
        b.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        let mut slot=b.residency.try_borrow_mut().map_err(|_|self.failure(Cause::Spent))?;
        if slot.is_some() || b.discover_attempted.get(){return Err(self.failure(Cause::Spent));}
        *slot=Some(Box::new(RegisteredResidency{access,attempt}));
        Ok(())
    }
    fn validate_demand(&self,demands:&IndexedDemandSource)->Result<(),Error> {
        let b=self.body();
        let completed=demands.source().and_then(|source|source.downcast_ref::<CompletedIds>())
            .ok_or_else(||self.failure(Cause::Identity))?;
        if !completed.source.same_owner(&b.identity) || !b.discovered.get() || !b.remapped.get()
            || b.closed.get() || !demands.funding().is_some_and(|funding|funding.same_account(&b.funding)) {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }
    pub(crate) fn with_demand_loan<R,E,F>(&self,demands:&IndexedDemandSource,run:F)
        ->Result<Result<R,E>,Error>
    where F:FnOnce(Option<PreparedIndexedDemandLoan<'_>>)->Result<R,E> {
        self.validate_demand(demands)?;
        let b=self.body();
        let frames=[size_of::<F>(),size_of::<R>(),size_of::<E>(),size_of::<Result<R,E>>(),
            size_of::<Result<Result<R,E>,Error>>(),size_of::<PreparedIndexedDemandLoan<'_>>(),
            size_of::<(&Self,&IndexedDemandSource)>(),
            eredu_nn::Error::retained_source_control_bytes::<Failure>().ok_or_else(||self.failure(Cause::Overflow))?];
        b.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        if b.residency.try_borrow().map_err(|_|self.failure(Cause::Spent))?.is_none(){return Err(self.failure(Cause::Identity));}
        Ok(run(Some(PreparedIndexedDemandLoan::new(self,&b.funding))))
    }
    pub(crate) fn acquire_from_demand(&self,request:ParameterBankAcquisition<'_>,demands:&IndexedDemandSource,
        funding:&WorkspaceMetadataFunding,stream:&Stream)->Result<AcquiredParameterGroups,Error> {
        self.validate_parent(stream)?;self.validate_demand(demands)?;
        let b=self.body();
        if request.access()!=b.identity.census.access() || b.route_copies.get()!=2 || !funding.same_account(&b.funding){return Err(self.failure(Cause::Identity));}
        let mut source=b.residency.try_borrow_mut().map_err(|_|self.failure(Cause::Spent))?;
        source.as_mut().ok_or_else(||self.failure(Cause::Identity))?
            .acquire(self,demands,request.entries(),stream)
            .map_err(|cause|self.failure(Cause::Consumer(cause)))
    }
    pub(crate) fn copy_route_value(&self,value:&MlxTensor,demands:&IndexedDemandSource,stream:&Stream)
        ->Result<MlxTensor,Error> {
        self.validate_parent(stream)?;self.validate_demand(demands)?;
        let b=self.body();
        let frames=[size_of::<(&MlxTensor,&IndexedDemandSource,&Stream)>(),size_of::<Result<MlxTensor,Error>>(),
            size_of::<safemlx::PreparedArrayClone>(),size_of::<Result<Array,safemlx::PreparedArrayCloneCause>>(),
            Array::inspection_clone_handle_bytes(),safemlx::PreparedArrayClone::control_bytes().ok_or_else(||self.failure(Cause::Overflow))?,
            Array::descriptor_comparison_control_bytes().ok_or_else(||self.failure(Cause::Overflow))?];
        b.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        if b.route_copies.get()>=2 || b.acquire_attempted.get() || value.shape()!=[
            i32::try_from(b.identity.census.rows()).map_err(|_|self.failure(Cause::Geometry))?,
            i32::try_from(b.identity.census.routes()).map_err(|_|self.failure(Cause::Geometry))?] {
            return Err(self.failure(Cause::Geometry));
        }
        b.route_copies.set(b.route_copies.get()+1);
        let mut slot=safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(|cause|self.failure(Cause::Clone(cause)))?;
        slot.fill_for_inspection(value.as_array()).map(MlxTensor::from_array).map_err(|cause|self.failure(Cause::Clone(cause)))
    }
}

// The dynamic window cannot exist until the completed IDs select its exact
// ordered roots. The registered access and finite bank arrive before discovery;
// the selected window is retained here through compact binding/completion.
struct SelectedResidency {
    attempt:Option<OriginalSelectedResidencyAttempt>,
    disk:Option<(eredu_runtime::working_memory::WorkingMemoryPool,
        crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity)>,
    bank:eredu_runtime::working_memory::OriginalHostSourceBank,
    access:OriginalSelectedResidencyAccess,
}
impl IndexedResidency for SelectedResidency {
    fn acquire(&mut self,source:&OriginalIndexedChunkSource,demands:&IndexedDemandSource,
        entries:&[(ParameterBankKey,u64)],stream:&Stream)->Result<AcquiredParameterGroups,Error> {
        if self.attempt.is_some(){return Err(source.failure(Cause::Spent));}
        let b=source.body();
        let frames=[size_of::<Vec<OffloadUnitId>>(),size_of::<OriginalSelectedResidencyAttempt>(),
            size_of::<Result<OriginalSelectedResidencyAttempt,Error>>(),
            size_of::<(ResidencyManager,crate::backend::runtime::residency::manager::SupplementaryResidencySource)>(),
            size_of::<(ParameterBankKey,u64)>(),size_of::<[u8;ParameterBankKey::unit_id_buffer_bytes()]>(),
            Layout::array::<OffloadUnitId>(entries.len()).map_err(|_|source.failure(Cause::Overflow))?.size()];
        let bytes=entries.iter().try_fold(0usize,|bytes,(key,_)|bytes.checked_add(key.unit_id_length()))
            .and_then(|bytes|frames.into_iter().try_fold(bytes.checked_add(size_of_val(&frames))?,usize::checked_add))
            .ok_or_else(||source.failure(Cause::Overflow))?;
        b.funding.reserve_metadata(bytes).map_err(|cause|source.failure(Cause::Funding(cause)))?;
        let (manager,selected)=b.identity.bank.with_workspace_source(&b.funding,|actual| {
            if !actual.same_source(&b.identity.bank) || actual.parameter_revision()!=b.identity.parameter_revision {
                return Err(source.failure(Cause::Identity));
            }
            let manager=actual.manager();
            let selected=manager.supplementary_residency_source().ok_or_else(||source.failure(Cause::Identity))?;
            Ok::<_,Error>((manager.clone(),selected.clone()))
        }).map_err(|cause|source.failure(Cause::Bank(cause)))??;
        let mut roots=Vec::new();
        roots.try_reserve_exact(entries.len()).map_err(|cause|source.failure(Cause::Allocation(cause)))?;
        for (key,_) in entries { roots.push(key.unit_id()); }
        self.attempt=Some(self.access.prepare(&manager,&selected,&roots,&mut self.bank,
            self.disk.as_ref().map(|(pool,capacity)|(pool,capacity,&b.funding)))
            .map_err(|cause|source.failure(Cause::Consumer(cause)))?);
        self.access.with_residency(self.attempt.as_mut().ok_or_else(||source.failure(Cause::Spent))?,|slots,observer| {
            if !observer.same_scope(&b.observer){return Err(source.failure(Cause::Identity));}
            source.acquire_from_slots(demands,entries,slots,stream)
        })
    }
}
impl OriginalIndexedChunkSource {
    /// Installs the real operation and its finite selected-window constructor
    /// bank before ID discovery. No static layerwise ordinal is substituted.
    pub(crate) fn bind_selected_residency<U:'static>(&self,access:OriginalOperationAccess<U>,
        bank:eredu_runtime::working_memory::OriginalHostSourceBank,stream:&Stream)->Result<(),Error> {
        self.bind_selected_residency_source(access.selected_residency_access()
            .map_err(|cause|self.failure(Cause::Consumer(cause)))?,bank,None,stream)
    }
    /// The selected foreground read capacity is retained alongside the actual
    /// operation and cannot be recreated or reset by a demand result.
    pub(crate) fn bind_selected_disk_residency<U:'static>(&self,access:OriginalOperationAccess<U>,
        bank:eredu_runtime::working_memory::OriginalHostSourceBank,
        pool:eredu_runtime::working_memory::WorkingMemoryPool,
        capacity:crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity,
        stream:&Stream)->Result<(),Error> {
        self.bind_selected_residency_source(access.selected_residency_access()
            .map_err(|cause|self.failure(Cause::Consumer(cause)))?,bank,Some((pool,capacity)),stream)
    }
    pub(crate) fn bind_selected_residency_source(&self,access:OriginalSelectedResidencyAccess,
        bank:eredu_runtime::working_memory::OriginalHostSourceBank,
        disk:Option<(eredu_runtime::working_memory::WorkingMemoryPool,
            crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity)>,stream:&Stream)->Result<(),Error> {
        self.validate_parent(stream)?;
        let b=self.body();
        access.validate_observer(&b.observer).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        access.validate_bank(&bank).map_err(|cause|self.failure(Cause::Consumer(cause)))?;
        let frames=[size_of::<SelectedResidency>(),Layout::new::<SelectedResidency>().size(),
            size_of::<Box<SelectedResidency>>(),size_of::<Box<dyn IndexedResidency>>(),
            size_of::<Option<Box<dyn IndexedResidency>>>(),size_of::<OriginalSelectedResidencyAccess>(),
            size_of::<eredu_runtime::working_memory::OriginalHostSourceBank>(),
            size_of::<std::cell::RefMut<'_,Option<Box<dyn IndexedResidency>>>>(),size_of::<Result<(),Error>>(),
            OriginalSelectedResidencyAccess::control_bytes().ok_or_else(||self.failure(Cause::Overflow))?];
        b.funding.reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
            .ok_or_else(||self.failure(Cause::Overflow))?).map_err(|cause|self.failure(Cause::Funding(cause)))?;
        let mut slot=b.residency.try_borrow_mut().map_err(|_|self.failure(Cause::Spent))?;
        if slot.is_some() || b.discover_attempted.get(){return Err(self.failure(Cause::Spent));}
        *slot=Some(Box::new(SelectedResidency{attempt:None,disk,bank,access}));
        Ok(())
    }
}
