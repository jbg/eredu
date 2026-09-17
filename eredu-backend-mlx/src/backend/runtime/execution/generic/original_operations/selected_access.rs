//! Exact selected-residency authority shared by registered and resident roles.
use super::*;
use crate::backend::runtime::residency::manager::{SupplementaryResidencySource,ForegroundDiskSourceCapacity,OriginalResidencySlots};
use eredu_runtime::working_memory::{OriginalHostSourceBank,OriginalHostSourceCustody,WorkingMemoryPool,WorkingMemoryReservation};
use eredu_nn::workspace::WorkspaceMetadataFunding;

#[derive(Clone)]
enum Source {
    Registered {registry:Rc<Registry>,controls:OperationControls},
    Resident(resident::SelectedResidencyProjection),
}
/// Retains the exact installed role; it contains no prepared storage, bank
/// grant, native graph or family policy. A stale source always refuses.
#[derive(Clone)]
pub(crate) struct OriginalSelectedResidencyAccess {source:Source,nested:Option<NestedScope>}
#[derive(Clone)]
struct NestedScope {parent:safemlx::OriginalScopeObserver,child:safemlx::OriginalScopeObserver}
/// Authenticated while the enclosing source is still the active native role.
/// Binding requires the child relation created by the shared Recovery worker.
pub(crate) struct PreparedSelectedResidencyAccess {
    access:OriginalSelectedResidencyAccess,
    parent:safemlx::OriginalScopeObserver,
    funding:WorkspaceMetadataFunding,
}
impl PreparedSelectedResidencyAccess {
    pub(crate) fn bind_control_bytes()->Option<usize> {
        let frames=[size_of::<OriginalSelectedResidencyAccess>(),size_of::<NestedScope>(),
            size_of::<Result<OriginalSelectedResidencyAccess,Error>>(),
            safemlx::OriginalScopeObserver::control_bytes()?.checked_mul(2)?];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
    }
    pub(crate) fn bind(&self,context:&crate::backend::submission_recovery::native_role::NativeRoleContext<'_>)
        ->Result<OriginalSelectedResidencyAccess,Error> {
        self.funding.reserve_metadata(Self::bind_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if !context.has_parent(&self.parent) || self.access.nested.is_some(){return Err(identity());}
        self.access.authenticate_observer(&self.parent)?;
        let access=OriginalSelectedResidencyAccess{source:self.access.source.clone(),
            nested:Some(NestedScope{parent:self.parent.clone(),child:context.observer().clone()})};
        access.authenticate()?;
        Ok(access)
    }
}
impl OriginalSelectedResidencyAccess {
    pub(super) fn registered(registry:Rc<Registry>,controls:OperationControls)->Result<Self,Error> {
        controls.validate(&registry)?;
        registry.authenticate()?;
        Ok(Self{source:Source::Registered{registry,controls},nested:None})
    }
    pub(super) fn resident(source:resident::SelectedResidencyProjection)->Result<Self,Error> {
        source.authenticate()?;
        Ok(Self{source:Source::Resident(source),nested:None})
    }
    fn authenticate(&self)->Result<safemlx::OriginalScopeObserver,Error> {
        let observer=safemlx::OriginalScopeObserver::require_current()?;
        if let Some(nested)=&self.nested {
            if !nested.child.same_scope(&observer){return Err(identity());}
            self.authenticate_observer(&nested.parent)?;
        }else{self.authenticate_observer(&observer)?;}
        if let Some(cause)=observer.retained_failure(){return Err(cause.into());}
        if observer.status().blocked(){return Err(identity());}
        Ok(observer)
    }
    fn authenticate_observer(&self,observer:&safemlx::OriginalScopeObserver)->Result<(),Error> {
        match &self.source {
            Source::Registered{registry,controls}=>{controls.validate(registry)?;registry.authenticate_observer(observer)},
            Source::Resident(source)=>source.authenticate_observer(observer),
        }
    }
    pub(crate) fn native_role_control_bytes()->Option<usize> {
        let frames=[size_of::<PreparedSelectedResidencyAccess>(),size_of::<Result<PreparedSelectedResidencyAccess,Error>>(),
            safemlx::OriginalScopeObserver::control_bytes()?];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
    }
    pub(crate) fn prepare_native_role(&self,funding:&WorkspaceMetadataFunding)->Result<PreparedSelectedResidencyAccess,Error> {
        funding.reserve_metadata(Self::native_role_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if self.nested.is_some(){return Err(identity());}
        Ok(PreparedSelectedResidencyAccess{access:self.clone(),parent:self.authenticate()?,funding:funding.clone()})
    }
    fn same_source(&self,other:&Self)->bool {
        let scope=match (&self.nested,&other.nested) {
            (None,None)=>true,
            (Some(a),Some(b))=>a.parent.same_scope(&b.parent)&&a.child.same_scope(&b.child),
            _=>false,
        };
        if !scope{return false;}
        match (&self.source,&other.source) {
            (Source::Registered{registry:a,..},Source::Registered{registry:b,..})=>Rc::ptr_eq(a,b),
            (Source::Resident(a),Source::Resident(b))=>a.same_source(b),
            _=>false,
        }
    }
    fn reservation(&self)->Option<&WorkingMemoryReservation> {
        match &self.source {Source::Registered{registry,..}=>registry.request.memory_reservation(),Source::Resident(_)=>None}
    }
    fn custody(&self)->OriginalHostSourceCustody {
        match &self.source {
            Source::Registered{controls:OperationControls::Text(controls),..}=>controls.clone().into(),
            Source::Registered{controls:OperationControls::Speculative(role),..}=>role.budget_custody().into(),
            Source::Registered{controls:OperationControls::Realtime(custody),..}=>custody.clone().into(),
            Source::Resident(source)=>source.custody(),
        }
    }
    /// Same accepted metadata account as the currently authenticated source.
    /// This is a lifetime alias only; all source-bank extraction stays checked.
    pub(crate) fn metadata_custody(&self)->Result<eredu_runtime::working_memory::OriginalOperationMetadataCustody,Error> {
        self.authenticate()?;
        Ok(self.custody().metadata_custody())
    }
    pub(crate) fn validate_observer(&self,observer:&safemlx::OriginalScopeObserver)->Result<(),Error> {
        if !self.authenticate()?.same_scope(observer){return Err(identity());}Ok(())
    }
    /// Authenticates the actual root account before issuing exact direct children.
    /// The caller must still validate every selected child's finite facts.
    pub(crate) fn validate_source_bank(&self,bank:&OriginalHostSourceBank)->Result<(),Error> {
        self.authenticate()?;
        if !bank.belongs_to_source(&self.custody()){return Err(identity());}
        Ok(())
    }
    /// Validates the complete still-unspent accepted population before the
    /// shared invocation splits its once-only selected-window children.
    pub(crate) fn validate_population(&self,bank:&OriginalHostSourceBank,
        facts:eredu_runtime::working_memory::HostSourceConstructionFacts)->Result<(),Error> {
        self.authenticate()?;
        if !bank.belongs_to_source(&self.custody()) || !bank.matches_facts(facts){return Err(identity());}
        Ok(())
    }
    pub(crate) fn prepare_read_capacity(&self,
        source:&crate::backend::runtime::residency::manager::ForegroundDiskSubsetCeiling,
        calls:usize,bank:OriginalHostSourceBank,
    )->Result<ForegroundDiskSourceCapacity,Error> {
        self.authenticate()?;
        source.prepare_capacity(calls,bank,self.custody(),self.reservation()).map_err(memory)
    }
    /// Binds the one exact admitted reader source root before any region runs.
    pub(crate) fn prepare_read_series(&self,
        source:&crate::backend::runtime::residency::manager::ForegroundDiskSourceSeries,
        bank:OriginalHostSourceBank)->Result<ForegroundDiskSourceCapacity,Error> {
        self.authenticate()?;
        source.prepare_capacity(bank,self.custody(),self.reservation()).map_err(memory)
    }
    /// Each actual parent supplies its own observer/access loan. The read
    /// capacity itself is independent of observer lifetime and owns no scope.
    pub(crate) fn validate_read_capacity(&self,
        source:&crate::backend::runtime::residency::manager::ForegroundDiskSubsetCeiling,
        capacity:&ForegroundDiskSourceCapacity)->Result<(),Error> {
        self.authenticate()?;
        capacity.validate_account(self.reservation()).map_err(memory)?;
        if !capacity.custody().same_source(&self.custody())||!source.matches_capacity(capacity){return Err(identity());}
        Ok(())
    }
    pub(crate) fn validate_bank(&self,bank:&OriginalHostSourceBank)->Result<(),Error> {
        self.authenticate()?;
        if !bank.belongs_to_source(&self.custody()) || bank.remaining_attempts()!=OriginalSelectedResidencyAttempt::construction_attempts(){return Err(identity());}
        Ok(())
    }
    pub(crate) fn prepare(&self,manager:&ResidencyManager,source:&SupplementaryResidencySource,
        roots:&[OffloadUnitId],bank:&mut OriginalHostSourceBank,
        disk:Option<(&WorkingMemoryPool,&ForegroundDiskSourceCapacity,&WorkspaceMetadataFunding)>,
    )->Result<OriginalSelectedResidencyAttempt,Error> {
        self.validate_bank(bank)?;
        let value=storage::PreparedSelectedResidency::prepare_with_disk(manager,source,roots,bank,
            self.custody(),self.reservation(),disk)?;
        Ok(OriginalSelectedResidencyAttempt{value,access:self.clone()})
    }
    pub(crate) fn with_residency<T>(&self,attempt:&mut OriginalSelectedResidencyAttempt,
        execute:impl FnOnce(&mut OriginalResidencySlots<'_>,&safemlx::OriginalScopeObserver)->Result<T,Error>,
    )->Result<T,Error> {
        if !self.same_source(&attempt.access){return Err(identity());}
        let observer=self.authenticate()?;
        execute(&mut attempt.value.residency(self.reservation()),&observer)
    }
    pub(crate) fn control_bytes()->Option<usize> {
        let frames=[size_of::<Self>(),size_of::<NestedScope>(),size_of::<PreparedSelectedResidencyAccess>(),size_of::<Source>(),size_of::<Option<Self>>(),size_of::<Result<Self,Error>>(),
            size_of::<OperationControls>(),size_of::<OriginalHostSourceCustody>(),size_of::<(&Self,&OriginalHostSourceBank)>(),
            size_of::<(&Self,&safemlx::OriginalScopeObserver)>(),size_of::<Result<(),Error>>(),
            size_of::<Result<safemlx::OriginalScopeObserver,Error>>(),safemlx::OriginalScopeObserver::control_bytes()?];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames),usize::checked_add)
    }
}
