//! Independent native storage admission for one reached coordinator-word producer.
//! The model request has not necessarily been admitted. This uses the existing
//! cold source account/ledger, never a prepared-input B or model Q relabel.
use super::{original_prepared_native_input::Account,WorkingMemoryError,WorkingMemoryPool};
use crate::RetainedCommunicationSource;
use std::{fmt,mem::{size_of,size_of_val}};

/// Move-only storage custody for one accepted readiness producer. The concrete
/// producer retains it through native allocation, completion and quarantine.
/// This grants neither a model invocation nor another exchange or allocation.
#[derive(Debug)]
pub struct CommunicationPreparationCustody(Account);
impl CommunicationPreparationCustody {
    /// Check the existing exact domain without creating another reservation.
    pub fn validate_pool(&self,pool:&WorkingMemoryPool)->Result<(),WorkingMemoryError>{
        if self.0.matches_pool(pool){Ok(())}else{Err(WorkingMemoryError::IdentityMismatch)}
    }
}

/// One actual source-bound native producer used by the existing readiness
/// protocol: its immutable input materialization or submitted gather. Complete
/// source-derived native/backing/control populations must be known before this
/// method is called. Host planning alone is not this storage permission.
pub trait CommunicationPreparationProducer:Sized {
    /// Actual initialized input or owning submission/completion representation.
    type Output;
    /// Typed cause retaining every unretired native prefix independently.
    type Error:std::error::Error;
    /// Exact retained communication table/session source.
    fn source(&self)->&RetainedCommunicationSource;
    /// Actual finite word frame supplied by the shared neutral coordinator.
    fn frame(&self)->&[u32];
    /// Authenticate source and already initialized allocator in this same pool.
    fn validate_pool(&self,pool:&WorkingMemoryPool)->Result<(),WorkingMemoryError>;
    /// All managed native/backing/control storage born in this one producer.
    /// Independently retained host planning/source holds remain separate.
    fn required_storage_bytes(&self)->Result<usize,WorkingMemoryError>;
    /// Optional enclosing operation policy. This can only refuse; the same
    /// independent source/domain admission below remains required.
    fn check_preparation_policy(&self, _pool: &WorkingMemoryPool, _bytes: u64) -> Result<(), WorkingMemoryError> { Ok(()) }
    /// Execute once after comparison. Native aliases must keep the supplied
    /// custody independently of the returned wrapper, also on failure/unwind.
    fn produce(self,custody:CommunicationPreparationCustody)->Result<Self::Output,Self::Error>;
}

/// One accepted producer output and its independent original storage hold.
/// Mutable access is lexical; no raw account or new amount can be extracted.
pub struct PreparedCommunicationPreparation<T> {
    output:T,
    source:RetainedCommunicationSource,
    account:Account,
}
impl<T> PreparedCommunicationPreparation<T> {
    /// Borrow the actual initialized input or completion without cloning it.
    pub fn output(&self)->&T{&self.output}
    /// Lend mutable mechanism state for existing completion/extraction workers.
    pub fn with_output<R>(&mut self,run:impl FnOnce(&mut T)->R)->R{run(&mut self.output)}
    /// Immutable source identity retained through every native exit.
    pub fn source(&self)->&RetainedCommunicationSource{&self.source}
    /// Validate this accepted storage's original domain.
    pub fn validate_pool(&self,pool:&WorkingMemoryPool)->Result<(),WorkingMemoryError>{
        if self.account.matches_pool(pool){Ok(())}else{Err(WorkingMemoryError::IdentityMismatch)}
    }
}
impl<T> fmt::Debug for PreparedCommunicationPreparation<T> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("PreparedCommunicationPreparation").field("original_bytes",&self.account.held_bytes()).finish_non_exhaustive()
    }
}

/// Rejection or producer failure. Any uncalled plan/completed prefix retires
/// before the exact retained source and independent native storage account.
pub struct CommunicationPreparationError<P:CommunicationPreparationProducer> {
    accounting:Option<WorkingMemoryError>,
    production:Option<P::Error>,
    output:Option<P::Output>,
    plan:Option<P>,
    source:RetainedCommunicationSource,
    account:Option<Account>,
}
impl<P:CommunicationPreparationProducer> CommunicationPreparationError<P> {
    fn rejected(plan:P,cause:WorkingMemoryError)->Self {
        let source=plan.source().clone();
        Self{accounting:Some(cause),production:None,output:None,plan:Some(plan),source,account:None}
    }
    /// Retire the concrete prefix through its existing guarded Drop path. Native
    /// quarantine retains its own account; no completion or refund is inferred.
    pub fn retire(self)->RetiredCommunicationPreparationError<P::Error>{
        let Self{accounting,production,output,plan,source,account}=self;
        drop(output);drop(plan);
        RetiredCommunicationPreparationError{accounting,production,source,account}
    }
    /// Exact fixed move/destruction controls for the owning retirement adapter.
    pub fn retirement_control_bytes()->Option<usize>{
        let parts=[size_of::<Self>(),size_of::<Option<P::Output>>(),size_of::<Option<P>>(),
            size_of::<RetiredCommunicationPreparationError<P::Error>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
}
impl<P:CommunicationPreparationProducer> fmt::Debug for CommunicationPreparationError<P> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("CommunicationPreparationError").field("accounting",&self.accounting)
            .field("production",&self.production).finish_non_exhaustive()
    }
}
impl<P:CommunicationPreparationProducer> fmt::Display for CommunicationPreparationError<P> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        if let Some(cause)=&self.production{fmt::Display::fmt(cause,f)}else{fmt::Display::fmt(self.accounting.as_ref().expect("refused producer"),f)}
    }
}
impl<P:CommunicationPreparationProducer> std::error::Error for CommunicationPreparationError<P>
where P::Error:'static {
    fn source(&self)->Option<&(dyn std::error::Error+'static)>{
        self.production.as_ref().map(|cause|cause as &(dyn std::error::Error+'static))
            .or_else(||self.accounting.as_ref().map(|cause|cause as &(dyn std::error::Error+'static)))
    }
}
/// Typed original cause plus source/account after native prefix retirement.
#[derive(Debug)]
pub struct RetiredCommunicationPreparationError<E> {
    accounting:Option<WorkingMemoryError>,
    production:Option<E>,
    source:RetainedCommunicationSource,
    account:Option<Account>,
}
impl<E:std::error::Error> fmt::Display for RetiredCommunicationPreparationError<E> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        if let Some(cause)=&self.production{fmt::Display::fmt(cause,f)}else{fmt::Display::fmt(self.accounting.as_ref().expect("refused producer"),f)}
    }
}
impl<E:std::error::Error+'static> std::error::Error for RetiredCommunicationPreparationError<E> {
    fn source(&self)->Option<&(dyn std::error::Error+'static)>{
        self.production.as_ref().map(|cause|cause as &(dyn std::error::Error+'static))
            .or_else(||self.accounting.as_ref().map(|cause|cause as &(dyn std::error::Error+'static)))
    }
}
impl WorkingMemoryPool {
    /// Closed producer contribution plus existing cold account/result controls.
    /// No scalar byte/graph grant or native resource is created by this query.
    pub fn communication_preparation_required_bytes<P:CommunicationPreparationProducer>(plan:&P)
        ->Result<u64,WorkingMemoryError> {
        let controls=[Account::storage_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<Account>(),size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<CommunicationPreparationCustody>(),size_of::<P>(),
            size_of::<RetainedCommunicationSource>(),size_of::<&[u32]>(),
            size_of::<Result<P::Output,P::Error>>(),size_of::<Result<(),WorkingMemoryError>>(),
            size_of::<PreparedCommunicationPreparation<P::Output>>(),size_of::<CommunicationPreparationError<P>>(),
            size_of::<Result<PreparedCommunicationPreparation<P::Output>,CommunicationPreparationError<P>>>(),
            CommunicationPreparationError::<P>::retirement_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
        ];
        let bytes=controls.into_iter().try_fold(plan.required_storage_bytes()?,usize::checked_add)
            .and_then(|n|n.checked_add(size_of_val(&controls))).ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_|WorkingMemoryError::Overflow)
    }
    /// One real readiness producer, admitted independently before the later
    /// model quote. Uses the same source ledger and account handoff as existing
    /// native source construction; each reached attempt is compared separately.
    pub fn prepare_communication<P:CommunicationPreparationProducer>(&self,plan:P)
        ->Result<PreparedCommunicationPreparation<P::Output>,CommunicationPreparationError<P>> {
        if let Err(cause)=plan.validate_pool(self){return Err(CommunicationPreparationError::rejected(plan,cause));}
        let bytes=match Self::communication_preparation_required_bytes(&plan){
            Ok(bytes)=>bytes,Err(cause)=>return Err(CommunicationPreparationError::rejected(plan,cause)),
        };
        if let Err(cause)=plan.check_preparation_policy(self,bytes){return Err(CommunicationPreparationError::rejected(plan,cause));}
        let allowance=match self.admit_source_compiler(bytes){
            Ok(value)=>value,Err(cause)=>return Err(CommunicationPreparationError::rejected(plan,cause)),
        };
        let account=allowance.into_prepared_native_account();
        let source=plan.source().clone();
        let result=plan.produce(CommunicationPreparationCustody(account.share()));
        let accounting=account.finish().err();
        match result {
            Ok(output) if accounting.is_none()=>Ok(PreparedCommunicationPreparation{output,source,account}),
            Ok(output)=>Err(CommunicationPreparationError{accounting,production:None,output:Some(output),plan:None,source,account:Some(account)}),
            Err(cause)=>Err(CommunicationPreparationError{accounting,production:Some(cause),output:None,plan:None,source,account:Some(account)}),
        }
    }
}
