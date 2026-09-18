use super::*;
use eredu_nn::workspace::*;
use std::{convert::Infallible,sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}}};
#[derive(Debug,Default)]
struct State {calls:AtomicUsize,refuse:AtomicBool,cut:AtomicUsize,retired:AtomicBool}
#[derive(Debug)]struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self,bytes:usize)->Result<(),HostMetadataFundingError>{
        let ordinal=self.0.calls.fetch_add(1,Ordering::SeqCst)+1;
        let cut=self.0.cut.load(Ordering::SeqCst);
        if self.0.refuse.load(Ordering::SeqCst)||(cut!=0&&ordinal>=cut) {Err(HostMetadataFundingError::Capacity{required:bytes as u64,available:0})}else{Ok(())}
    }
}
impl Drop for Account {fn drop(&mut self){self.0.retired.store(true,Ordering::SeqCst);}}
#[derive(Debug)]struct Missing;
impl WorkspaceMechanisms for Missing {
    fn operation_bound(&self,_:&WorkspaceOperation)->Result<Option<WorkspaceOperationBound>,Error>{panic!("finite source expected")}
}
impl WorkspaceFactMechanisms for Missing {
    type Error=Infallible;
    fn operation_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{Ok(None)}
    fn write_operation_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceEffectDestination<'_>)->Result<Option<WorkspaceOperationFacts>,Infallible>{panic!("missing source has no output destination")}
    fn host_facts(&self,_:WorkspaceOperationView<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{Ok(None)}
    fn write_host_facts(&self,_:WorkspaceOperationView<'_>,_:WorkspaceHostDestination<'_>)->Result<Option<WorkspaceHostFacts>,Infallible>{panic!("missing source has no host destination")}
}
#[derive(Debug,thiserror::Error)]#[error("original unit cause")]struct Original;
fn fixture()->(WorkspaceContext,Arc<State>){
    let state=Arc::new(State::default());
    let context=WorkspaceContext::new_with_metadata_funding(Missing,HostMetadataFunding::new(Account(state.clone())).unwrap()).unwrap();
    (context,state)
}
#[test]
fn failed_unit_moves_first_missing_equation_and_retains_its_actual_account(){
    use std::error::Error as _;
    let (context,state)=fixture();
    let permit=UnitFailure::prepare(Some(&context)).unwrap();
    let value=context.execute(WorkspaceOperationKind::Elementwise("missing source"),&[],vec![context.layout(&[2],WorkspaceDtype::Float32).unwrap()]).unwrap();
    let cause=context.metadata_source(Original);
    let error=permit.retain(cause);
    let diagnostic=error.source().unwrap().downcast_ref::<StageFailure>().unwrap();
    let (index,operation)=diagnostic.equation.as_ref().unwrap();
    assert_eq!(*index,0);assert!(operation.contains("missing source"));
    assert!(diagnostic.inspection.is_none());assert!(diagnostic.cause.source().unwrap().is::<Original>());
    drop(value);drop(context);assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);assert!(state.retired.load(Ordering::SeqCst));
}
#[test]
fn failed_unit_keeps_original_and_inspection_refusal_without_retrying_account(){
    use std::error::Error as _;
    let (context,state)=fixture();
    let permit=UnitFailure::prepare(Some(&context)).unwrap();
    let cause=context.metadata_source(Original);
    state.refuse.store(true,Ordering::SeqCst);
    let before=state.calls.load(Ordering::SeqCst);
    let error=permit.retain(cause);
    assert_eq!(state.calls.load(Ordering::SeqCst),before+1);
    let diagnostic=error.source().unwrap().downcast_ref::<StageFailure>().unwrap();
    assert!(diagnostic.cause.source().unwrap().is::<Original>());
    assert!(diagnostic.inspection.as_ref().unwrap().source().unwrap().is::<WorkspaceMetadataError>());
    assert!(diagnostic.equation.is_none());
    drop(context);assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);assert!(state.retired.load(Ordering::SeqCst));
}
#[test]
fn existing_unit_funding_refusal_does_not_start_inspection(){
    use std::error::Error as _;
    let (context,state)=fixture();
    let permit=UnitFailure::prepare(Some(&context)).unwrap();
    state.refuse.store(true,Ordering::SeqCst);
    let cause=Error::from(WorkspaceMetadataError::Funding(HostMetadataFundingError::Unavailable));
    let before=state.calls.load(Ordering::SeqCst);
    let error=permit.retain(cause);
    assert_eq!(state.calls.load(Ordering::SeqCst),before);
    assert!(error.source().unwrap().is::<WorkspaceMetadataError>());
}

#[test]
fn aborted_quote_keeps_both_causes_at_every_reached_inspection_refusal(){
    use std::error::Error as _;
    let run=|cut:Option<usize>|{
        let (context,state)=fixture();
        let permit=UnitFailure::prepare(Some(&context)).unwrap();
        let values=context.execute(WorkspaceOperationKind::Elementwise("missing source"),&[],
            vec![context.layout(&[2],WorkspaceDtype::Float32).unwrap()]).unwrap();
        let cause=context.metadata_source(Original);
        let before=state.calls.load(Ordering::SeqCst);
        if let Some(cut)=cut {state.cut.store(before+cut,Ordering::SeqCst);}
        let error=permit.retain(cause);
        let calls=state.calls.load(Ordering::SeqCst)-before;
        let diagnostic=error.source().unwrap().downcast_ref::<StageFailure>().unwrap();
        assert!(diagnostic.cause.source().unwrap().is::<Original>());
        match cut {
            Some(cut)=>{
                assert_eq!(calls,cut,"no account call follows the first refusal");
                assert!(diagnostic.equation.is_none());
                let refusal=diagnostic.inspection.as_ref().expect("inspection refusal retained");
                assert!(matches!(refusal.source().unwrap().downcast_ref::<WorkspaceMetadataError>(),
                    Some(WorkspaceMetadataError::Funding(HostMetadataFundingError::Capacity{available:0,..}))));
            }
            None=>{assert!(diagnostic.equation.is_some());assert!(diagnostic.inspection.is_none());}
        }
        drop(values);drop(context);assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);assert!(state.retired.load(Ordering::SeqCst));
        calls
    };
    let reached=run(None);
    assert!(reached>=3,"report and descriptor producers must both be reached");
    for cut in 1..=reached {run(Some(cut));}
}

#[test]
fn missing_floating_output_takes_precedence_over_an_earlier_zero_output_operation(){
    use std::error::Error as _;
    let (context,state)=fixture();
    let permit=UnitFailure::prepare(Some(&context)).unwrap();
    context.execute(WorkspaceOperationKind::ValueCompletion,&[],vec![]).unwrap();
    let values=context.execute(WorkspaceOperationKind::Elementwise("lost scalar"),&[],
        vec![context.layout(&[2],WorkspaceDtype::Float32).unwrap()]).unwrap();
    let error=permit.retain(context.metadata_source(Original));
    let diagnostic=error.source().unwrap().downcast_ref::<StageFailure>().unwrap();
    let (index,operation)=diagnostic.equation.as_ref().unwrap();
    assert_eq!(*index,1);assert!(operation.contains("lost scalar"));
    assert!(diagnostic.cause.source().unwrap().is::<Original>());
    drop(values);drop(context);assert!(!state.retired.load(Ordering::SeqCst));
    drop(error);assert!(state.retired.load(Ordering::SeqCst));
}
