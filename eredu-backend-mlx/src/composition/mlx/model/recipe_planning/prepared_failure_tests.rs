use super::*;
use eredu_nn::workspace::HostMetadataAccount;
use eredu_runtime::working_memory::{WorkingMemoryError, SpeculativeRequestError};
use std::sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}};

#[derive(Debug)]
struct State {remaining:AtomicUsize,calls:AtomicUsize,retired:AtomicBool}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
        self.0.calls.fetch_add(1,Ordering::SeqCst);
        let remaining=self.0.remaining.load(Ordering::SeqCst);
        let next=remaining.checked_sub(bytes).ok_or(HostMetadataFundingError::Capacity{
            required:bytes as u64,available:remaining as u64})?;
        self.0.remaining.store(next,Ordering::SeqCst);
        Ok(())
    }
}
impl Drop for Account {fn drop(&mut self){self.0.retired.store(true,Ordering::SeqCst);}}
fn funding()->(HostMetadataFunding,Arc<State>){
    let state=Arc::new(State{remaining:AtomicUsize::new(usize::MAX),calls:AtomicUsize::new(0),retired:AtomicBool::new(false)});
    (HostMetadataFunding::new(Account(state.clone())).unwrap(),state)
}
#[test]
fn prepaid_phase_failure_preserves_budget_source_without_a_second_debit(){
    let (funding,state)=funding();
    let prepared=PreparedPlanningError::<SpeculativeRequestError>::prepare(&funding).unwrap();
    let calls=state.calls.load(Ordering::SeqCst);
    state.remaining.store(0,Ordering::SeqCst);
    let error=prepared.retain(WorkingMemoryError::BudgetExceeded{required_bytes:179957231,available_bytes:2418011}.into());
    assert_eq!(state.calls.load(Ordering::SeqCst),calls);
    assert!(planning_error_has_funding::<SpeculativeRequestError>(&error,&funding));
    let mut source:Option<&(dyn StdError+'static)>=Some(&error);
    let mut budget=false;
    while let Some(cause)=source {
        if let Some(WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes})=cause.downcast_ref::<WorkingMemoryError>() {
            assert_eq!((*required_bytes,*available_bytes),(179957231,2418011));budget=true;
        }
        source=cause.source();
    }
    assert!(budget);
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);
    assert!(state.retired.load(Ordering::SeqCst));
}
#[test]
fn prepaid_phase_backend_failure_matches_only_its_actual_account(){
    let (funding,state)=funding();
    let prepared=PreparedPlanningError::<Error>::prepare(&funding).unwrap();
    let calls=state.calls.load(Ordering::SeqCst);state.remaining.store(0,Ordering::SeqCst);
    let error=prepared.retain(Error::PrefillScopeUnavailable);
    assert_eq!(state.calls.load(Ordering::SeqCst),calls);
    assert!(planning_error_has_funding::<Error>(&error,&funding));
    let (other,_)=self::funding();
    assert!(!planning_error_has_funding::<Error>(&error,&other));
    assert!(!planning_error_has_funding::<SpeculativeRequestError>(&error,&funding));
    drop(funding);assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);assert!(state.retired.load(Ordering::SeqCst));
}
