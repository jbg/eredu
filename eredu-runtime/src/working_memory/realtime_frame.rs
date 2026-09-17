//! One complete frame's original account, using the shared atomic pool ledger.
use super::{InferenceExecutionIdentity,WorkingMemoryPool,WorkingMemoryError,Usage,
    PreparedAccountCommit,HostSourceConstructionFacts,OriginalHostSourceBank,
    funding::{AccountTicket,PendingAccount,PendingOriginal,AccountNode},qualified_storage};
use crate::realtime_session::RealtimeFrameOccurrence;
use std::{mem::{size_of,size_of_val},sync::{Arc,atomic::{AtomicU64,Ordering}}};
use eredu_core::{HostMetadataAccount,HostMetadataFunding,HostMetadataFundingError};

/// One source-derived native phase within the same accepted frame account.
/// Separate preparation and execution phases retain distinct physical, graph
/// and record populations; neither can borrow the other's unused capacity.
#[derive(Debug, Clone, Copy)]
pub struct RealtimeNativeRequirements {
    physical:u64, graph:u64, record:u64,
}
impl RealtimeNativeRequirements {
    /// Complete actual native source bounds; missing facts fail before admission.
    pub fn new(physical:Option<u64>,graph:Option<u64>,record:Option<u64>)
        ->Result<Self,WorkingMemoryError> {
        let value=Self { physical:physical.ok_or(WorkingMemoryError::UnknownBound)?,
            graph:graph.ok_or(WorkingMemoryError::UnknownBound)?,
            record:record.ok_or(WorkingMemoryError::UnknownBound)? };
        value.required_bytes()?;Ok(value)
    }
    fn required_bytes(self)->Result<u64,WorkingMemoryError> {
        self.physical.checked_add(self.graph).and_then(|n|n.checked_add(self.record))
            .ok_or(WorkingMemoryError::Overflow)
    }
}

/// Complete source-derived domains for one actual scheduler frame. Missing
/// domains refuse before admission; no configured limit supplies a missing fact.
#[derive(Debug)]
pub struct RealtimeFrameRequirements {
    occurrence:RealtimeFrameOccurrence,
    physical:u64,
    state:u64,
    graph:u64,
    record:u64,
    controls:u64,
    sources:HostSourceConstructionFacts,
    preparation:Option<RealtimeNativeRequirements>,
}
impl RealtimeFrameRequirements {
    /// The selected backend quotes actual frame equations, state, native wrapper
    /// and source construction. These descriptive numbers issue no authority.
    #[allow(clippy::too_many_arguments)]
    pub fn new(occurrence:RealtimeFrameOccurrence,physical:Option<u64>,state:Option<u64>,
        graph:Option<u64>,record:Option<u64>,controls:Option<u64>,sources:HostSourceConstructionFacts)
        ->Result<Self,WorkingMemoryError> {
        let value=Self{occurrence,physical:physical.ok_or(WorkingMemoryError::UnknownBound)?,
            state:state.ok_or(WorkingMemoryError::UnknownBound)?,graph:graph.ok_or(WorkingMemoryError::UnknownBound)?,
            record:record.ok_or(WorkingMemoryError::UnknownBound)?,controls:controls.ok_or(WorkingMemoryError::UnknownBound)?,sources,preparation:None};
        value.required_bytes()?;Ok(value)
    }
    /// Adds the actual pre-branch native copy phase. Its controls join the same
    /// cumulative host counter; its retained backing is charged in addition to
    /// execution while the copied state remains alive. No claim is issued here.
    pub fn with_preparation(mut self,source:RealtimeNativeRequirements,controls:Option<u64>)
        ->Result<Self,WorkingMemoryError> {
        if self.preparation.is_some(){return Err(WorkingMemoryError::AlreadyStarted);}
        self.controls=self.controls.checked_add(controls.ok_or(WorkingMemoryError::UnknownBound)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        self.preparation=Some(source);self.required_bytes()?;Ok(self)
    }
    /// Full incremental charge, including account and actual metadata wrappers.
    pub fn required_bytes(&self)->Result<u64,WorkingMemoryError> {
        [self.physical,self.state,self.graph,self.record,self.controls,self.sources.protected_bytes(),
            account_control_bytes()?,funding_control_bytes()?,
            self.preparation.map(RealtimeNativeRequirements::required_bytes).transpose()?.unwrap_or(0)].into_iter()
            .try_fold(0u64,u64::checked_add).ok_or(WorkingMemoryError::Overflow)
    }
}
/// A failed accepted prefix retains its ticket until its own storage retires.
#[derive(Debug)]
pub struct RealtimeFrameAdmissionError {cause:WorkingMemoryError,ticket:Option<AccountTicket>}
impl std::fmt::Display for RealtimeFrameAdmissionError {
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result {self.cause.fmt(f)}
}
impl std::error::Error for RealtimeFrameAdmissionError {
    fn source(&self)->Option<&(dyn std::error::Error+'static)> {Some(&self.cause)}
}
impl From<WorkingMemoryError> for RealtimeFrameAdmissionError {
    fn from(cause:WorkingMemoryError)->Self {Self{cause,ticket:None}}
}
impl RealtimeFrameAdmissionError {
    /// Original fixed accounting failure.
    pub fn cause(&self)->&WorkingMemoryError {&self.cause}
}
#[derive(Debug)]
struct FrameCharge {
    occurrence:RealtimeFrameOccurrence,
    physical:u64,state:u64,graph:u64,record:u64,controls:u64,
    metadata_spent:AtomicU64,
    execution:InferenceExecutionIdentity,
    ticket:AccountTicket,
}
// No weak owner escapes. Deallocate the shared shell before the final ticket.
#[derive(Debug)]
struct FrameAccount(Option<Arc<FrameCharge>>);
impl FrameAccount {
    fn value(&self)->&FrameCharge {self.0.as_deref().expect("live realtime frame account")}
    fn same(&self,other:&Self)->bool {
        Arc::ptr_eq(self.0.as_ref().expect("live account"),other.0.as_ref().expect("live account"))
    }
}
impl Clone for FrameAccount {
    fn clone(&self)->Self {Self(Some(Arc::clone(self.0.as_ref().expect("live frame account"))))}
}
impl Drop for FrameAccount {
    fn drop(&mut self) {if let Some(owner)=self.0.take(){drop(Arc::into_inner(owner));}}
}
/// Unique original frame issuer. Restoring model state does not clone or reset
/// this issuer. Native and source claims are each consumed before construction.
#[derive(Debug)]
pub struct OriginalRealtimeFrame {
    account:FrameAccount,
    sources:Option<HostSourceConstructionFacts>,
    native_issued:bool,
    preparation:Option<RealtimeNativeRequirements>,
    preparation_required:bool,
    funding_issued:bool,
}
impl WorkingMemoryPool {
    /// Atomically compares a complete frame against every live domain owner,
    /// then publishes through the same original account ledger as text/speculation.
    pub fn reserve_realtime_frame(&self,execution:&InferenceExecutionIdentity,
        requirements:RealtimeFrameRequirements,capacity:u64,application_limit:Option<u64>)
        ->Result<OriginalRealtimeFrame,RealtimeFrameAdmissionError> {
        let bytes=requirements.required_bytes()?;
        if let Some(limit)=application_limit {
            if bytes>limit {return Err(WorkingMemoryError::BudgetExceeded{required_bytes:bytes,available_bytes:limit}.into());}
        }
        let controls=requirements.controls.checked_add(funding_control_bytes()?).ok_or(WorkingMemoryError::Overflow)?;
        let floor=controls.checked_add(account_control_bytes()?)
            .and_then(|v|v.checked_add(requirements.sources.protected_bytes())).ok_or(WorkingMemoryError::Overflow)?;
        let pending={
            let mut usage=self.0.usage.lock().map_err(|_|WorkingMemoryError::Poisoned)?;
            let commit=PreparedAccountCommit::prepare(self,execution,&usage,bytes,Some(capacity),&[])?;
            PendingAccount::accept(self,execution,&mut usage,commit,bytes,Some(capacity),floor)?
        };
        let ticket=pending.publish();
        if let Err(cause)=ticket.status(){return Err(RealtimeFrameAdmissionError{cause,ticket:Some(ticket)});}
        let RealtimeFrameRequirements{occurrence,physical,state,graph,record,controls:_,sources,preparation}=requirements;
        let account=FrameAccount(Some(Arc::new(FrameCharge{occurrence,physical,state,graph,record,controls,
            metadata_spent:AtomicU64::new(0),execution:execution.clone(),ticket})));
        Ok(OriginalRealtimeFrame{account,sources:Some(sources),native_issued:false,
            preparation,preparation_required:preparation.is_some(),funding_issued:false})
    }
}
impl OriginalRealtimeFrame {
    /// Authenticate the actual executable and scheduler occurrence before use.
    pub fn validate(&self,execution:&InferenceExecutionIdentity,occurrence:RealtimeFrameOccurrence)
        ->Result<(),WorkingMemoryError> {
        let value=self.account.value();value.ticket.status()?;
        if Arc::ptr_eq(&value.execution.0,&execution.0) && value.occurrence==occurrence {Ok(())}
        else {Err(WorkingMemoryError::IdentityMismatch)}
    }
    /// One optional preparation claim, consumed before any native construction.
    /// Its owner must establish native completion before returning the prepared
    /// scheduler item. Dropping or failing it cannot refund either phase.
    pub fn claim_preparation_native(&mut self)->Result<OriginalRealtimeNative,WorkingMemoryError> {
        self.account.value().ticket.status()?;
        let capacity=self.preparation.take().ok_or(WorkingMemoryError::AlreadyStarted)?;
        Ok(OriginalRealtimeNative{account:self.account.clone(),capacity})
    }
    /// One native execution claim; a failed preparation still consumes its claim.
    pub fn claim_native(&mut self)->Result<OriginalRealtimeNative,WorkingMemoryError> {
        self.account.value().ticket.status()?;
        if self.native_issued{return Err(WorkingMemoryError::AlreadyStarted);}
        if self.preparation_required && self.preparation.is_some(){return Err(WorkingMemoryError::AlreadyStarted);}
        self.native_issued=true;
        let source=self.account.value();
        let capacity=RealtimeNativeRequirements{physical:source.physical.checked_add(source.state)
            .ok_or(WorkingMemoryError::Overflow)?,graph:source.graph,record:source.record};
        Ok(OriginalRealtimeNative{account:self.account.clone(),capacity})
    }
    /// One actual source bank, accepted with this same original frame account.
    pub fn take_source_constructions(&mut self)->Result<OriginalHostSourceBank,WorkingMemoryError> {
        self.account.value().ticket.status()?;
        let facts=self.sources.take().ok_or(WorkingMemoryError::AlreadyStarted)?;
        Ok(OriginalHostSourceBank::new_with_custody(facts,None,self.budget_custody().into()))
    }
    /// One cumulative host account over this frame's actually quoted controls.
    /// Construction reserves its own measured Box/shared-shell bytes first.
    pub fn metadata_funding(&mut self)->Result<HostMetadataFunding,HostMetadataFundingError> {
        if self.funding_issued{return Err(HostMetadataFundingError::Unavailable);}
        self.funding_issued=true;
        HostMetadataFunding::new(FrameMetadata(self.budget_custody()))
    }
    /// Accounting-only alias for canonical outputs and exact completion retention.
    pub fn budget_custody(&self)->OriginalRealtimeBudgetCustody {
        OriginalRealtimeBudgetCustody{account:self.account.clone()}
    }
}
/// Move-only native preparation claim; its capacities came from the accepted
/// source quote and cannot be extracted again by cloning accounting custody.
#[derive(Debug)]
pub struct OriginalRealtimeNative {account:FrameAccount,capacity:RealtimeNativeRequirements}
impl OriginalRealtimeNative {
    /// Exact combined newly allocated equation and state physical backing.
    pub fn physical_bytes(&self)->u64 {self.capacity.physical}
    /// Exact original graph population.
    pub fn graph_bytes(&self)->u64 {self.capacity.graph}
    /// Exact original record population.
    pub fn record_bytes(&self)->u64 {self.capacity.record}
    /// Account retention only; creating an alias does not duplicate this claim.
    pub fn budget_custody(&self)->OriginalRealtimeBudgetCustody {
        OriginalRealtimeBudgetCustody{account:self.account.clone()}
    }
}
/// Frame accounting only. Native arrays, metadata and completion retain it
/// independently; no bank, claim, model owner or frame payload is reachable.
#[derive(Debug,Clone)]
pub struct OriginalRealtimeBudgetCustody {account:FrameAccount}
impl OriginalRealtimeBudgetCustody {
    /// Exact original accepted account, never equal geometry or capacity.
    pub fn same_account(&self,other:&Self)->bool {self.account.same(&other.account)}
    pub(in crate::working_memory) fn execution(&self)->&InferenceExecutionIdentity {&self.account.value().execution}
    pub(in crate::working_memory) fn account_id(&self)->u64 {self.account.value().ticket.id()}
    pub(in crate::working_memory) fn pool(&self)->&WorkingMemoryPool {self.account.value().ticket.pool()}
    pub(in crate::working_memory) fn quarantine(&self){self.account.value().ticket.quarantine();}
    pub(in crate::working_memory) fn validate_copy_source(&self,pool:&WorkingMemoryPool,usage:&Usage)
        ->Result<(),WorkingMemoryError> {
        if !self.pool().same_domain(pool){return Err(WorkingMemoryError::IdentityMismatch);}
        self.account.value().ticket.validate_in(usage)
    }
}
#[derive(Debug)]
struct FrameMetadata(OriginalRealtimeBudgetCustody);
impl HostMetadataAccount for FrameMetadata {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError> {
        let value=self.0.account.value();
        value.ticket.status().map_err(|_|HostMetadataFundingError::Unavailable)?;
        let bytes=u64::try_from(bytes).map_err(|_|HostMetadataFundingError::Overflow)?;
        value.metadata_spent.fetch_update(Ordering::AcqRel,Ordering::Acquire,|spent|
            spent.checked_add(bytes).filter(|next|*next<=value.controls))
            .map(|_|()).map_err(|spent|HostMetadataFundingError::Capacity{
                required:bytes,available:value.controls.saturating_sub(spent)})
    }
}
fn funding_control_bytes()->Result<u64,WorkingMemoryError> {
    HostMetadataFunding::constructor_bytes::<FrameMetadata>().and_then(|n|u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}
fn account_control_bytes()->Result<u64,WorkingMemoryError> {
    let parts=[size_of::<AccountNode>(),size_of::<AccountTicket>(),size_of::<PendingAccount>(),
        size_of::<PendingOriginal>(),size_of::<PreparedAccountCommit<'_>>(),size_of::<FrameAccount>(),
        size_of::<FrameCharge>(),size_of::<RealtimeFrameRequirements>(),size_of::<OriginalRealtimeFrame>(),
        size_of::<OriginalRealtimeNative>(),size_of::<OriginalRealtimeBudgetCustody>(),
        size_of::<RealtimeNativeRequirements>(),size_of::<Option<RealtimeNativeRequirements>>(),
        size_of::<RealtimeFrameAdmissionError>(),size_of::<Result<OriginalRealtimeFrame,RealtimeFrameAdmissionError>>(),
        size_of::<Result<OriginalRealtimeNative,WorkingMemoryError>>(),size_of::<Result<OriginalHostSourceBank,WorkingMemoryError>>(),
        size_of::<(&WorkingMemoryPool,&InferenceExecutionIdentity,RealtimeFrameRequirements,u64,Option<u64>)>(),
        size_of::<(&OriginalRealtimeFrame,&InferenceExecutionIdentity,RealtimeFrameOccurrence)>(),
        size_of::<(&FrameMetadata,usize)>(),size_of::<Result<u64,u64>>(),size_of::<(u64,u64,u64)>(),
        size_of::<Result<(),WorkingMemoryError>>(),size_of::<WorkingMemoryError>()];
    let stack=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
        .and_then(|n|u64::try_from(n).ok()).ok_or(WorkingMemoryError::Overflow)?;
    stack.checked_add(qualified_storage::shared_bytes::<FrameCharge>()?).ok_or(WorkingMemoryError::Overflow)
}
