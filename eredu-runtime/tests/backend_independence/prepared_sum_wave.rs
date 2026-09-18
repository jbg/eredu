//! The same whole-wave source dispatch owns paid validation and failure custody.
use super::*;
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::partitioned_execution::PartitionExecutionError;
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
thread_local! {
    static WAVES: Cell<usize> = const { Cell::new(0) };
    static INSPECTIONS: Cell<usize> = const { Cell::new(0) };
    static SOURCE: RefCell<Option<HostMetadataFunding>> = const { RefCell::new(None) };
    static FAIL_NATIVE: Cell<bool> = const { Cell::new(false) };
}
#[derive(Debug)]
struct Account { calls:Arc<AtomicUsize>, cut:Arc<AtomicUsize>, live:Arc<AtomicBool> }
impl eredu_core::HostMetadataAccount for Account {
    fn reserve_metadata(&self,_:usize)->Result<(),HostMetadataFundingError> {
        let call=self.calls.fetch_add(1,Ordering::SeqCst);
        if call>=self.cut.load(Ordering::SeqCst) { Err(HostMetadataFundingError::Unavailable) } else { Ok(()) }
    }
}
impl Drop for Account { fn drop(&mut self) { self.live.store(false,Ordering::SeqCst); } }
fn source()->(Arc<AtomicUsize>,Arc<AtomicUsize>,Arc<AtomicBool>) {
    let calls=Arc::new(AtomicUsize::new(0));let cut=Arc::new(AtomicUsize::new(usize::MAX));
    let live=Arc::new(AtomicBool::new(true));
    let funding=HostMetadataFunding::new(Account{calls:calls.clone(),cut:cut.clone(),live:live.clone()}).unwrap();
    SOURCE.with(|source| *source.borrow_mut()=Some(funding));
    WAVES.set(0);INSPECTIONS.set(0);FAIL_NATIVE.set(false);
    (calls,cut,live)
}
pub(super) fn complete<E,V>(values:&[FakeTensor],group:&PartitionCollectiveGroup,mut validate:V)
    ->Result<Option<Vec<FakeTensor>>,eredu_core::BackendFailure>
where E:std::error::Error+Send+Sync+'static, V:FnMut(&[FakeTensor],&HostMetadataFunding,bool)->Result<(),E> {
    let funding=SOURCE.with(|source|source.borrow().clone());
    let Some(funding)=funding else { return Ok(None); };
    funding.reserve_metadata(std::mem::size_of::<V>() +
        eredu_core::BackendFailure::source_retention_peak_bytes::<ValidationFailure<E>>().unwrap())
        .map_err(HostMetadataFundingError::into_backend_failure)?;
    validate(values,&funding,false).map_err(|cause|eredu_core::BackendFailure::from_error(ValidationFailure {cause, funding:funding.clone()}))?;
    WAVES.set(WAVES.get()+1);
    let start=group.trace.borrow().len();
    let mut submissions=Vec::new();
    for value in values {
        submissions.push(PartitionCollectiveBackend::all_reduce_sum(value.clone(),group,&()).unwrap());
        if FAIL_NATIVE.get() { return Err(eredu_core::BackendFailure::from_error(NativeFailure)); }
    }
    let outputs=submissions.into_iter().map(|submission| {
        assert_eq!(group.trace.borrow().len(),start+values.len(),"completion preceded a matching construction");
        submission.wait().unwrap()
    }).collect::<Vec<_>>();
    validate(&outputs,&funding,true).map_err(|cause|eredu_core::BackendFailure::from_error(ValidationFailure {cause, funding:funding.clone()}))?;
    Ok(Some(outputs))
}
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct ValidationFailure<E:std::error::Error+'static> {
    #[source] cause:E,
    funding:HostMetadataFunding,
}
#[derive(Debug,thiserror::Error)]
#[error("actual wave constructor failure")]
struct NativeFailure;
struct Metadata;
impl CommunicationTensorMetadata<PartitionCollectiveBackend> for Metadata {
    fn dtype(&self,_:&FakeTensor)->TensorDtype { TensorDtype::F32 }
    fn shape(&self,value:&FakeTensor)->Vec<usize> { vec![value.0.len()] }
    fn fixed_metadata_with_funding(&self,value:&FakeTensor,funding:&HostMetadataFunding)
        ->Result<Option<(TensorDtype,usize,Option<usize>)>,HostMetadataFundingError> {
        funding.reserve_metadata(std::mem::size_of::<(&FakeTensor,usize)>())?;
        INSPECTIONS.set(INSPECTIONS.get()+1);
        Ok(Some((TensorDtype::F32,1,Some(value.0.len()))))
    }
}
type Communication=PartitionCommunication<PartitionCollectiveBackend,PartitionCollectiveGroup,(),Metadata>;
fn communication()->(Communication,Rc<RefCell<Vec<PartitionCollectiveCall>>>,CollectiveGroupId) {
    let trace=Rc::new(RefCell::new(Vec::new()));let id=CollectiveGroupId::new(1);
    let requirement=CommunicationOperationRequirement::tensors(CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,1,4,None).unwrap(),true).unwrap();
    let descriptor=CommunicationGroupDescriptor::new(id,0,vec![0,1],Some(0),
        CommunicationGroupRequirements::new([requirement]).unwrap()).unwrap();
    let manifest=CommunicationManifest::new(2,0,vec![descriptor],vec![]).unwrap()
        .with_completion_policy(test_completion_policy());
    let group=PartitionCollectiveGroup {members:vec![0,1],local_rank:0,trace:trace.clone()};
    (Communication::new(manifest,vec![RealizedCommunicationGroup::new(id,group)],vec![],Metadata).unwrap(),trace,id)
}
#[test]
fn retained_sum_wave_dispatches_once_and_validates_all_inputs_before_source() {
    let (_,_,live)=source();let (communication,trace,id)=communication();
    let values=vec![FakeTensor(vec![3,4]),FakeTensor(vec![5]),FakeTensor(vec![6,7,8])];
    assert_eq!(communication.all_reduce_sum_wave(values.clone(),id,&(),Some(&())).unwrap(),values);
    assert_eq!(WAVES.get(),1);assert_eq!(INSPECTIONS.get(),6);
    assert_eq!(trace.borrow().len(),3,"retained results were submitted again");
    assert!(communication.all_reduce_sum_wave(vec![FakeTensor(vec![1]),FakeTensor(vec![1;5])],id,&(),Some(&())).is_err());
    assert_eq!(trace.borrow().len(),3,"valid prefix submitted before later input validation");
    assert_eq!(WAVES.get(),1);
    SOURCE.with(|source|source.borrow_mut().take());assert!(!live.load(Ordering::SeqCst));
}
#[test]
fn retained_sum_wave_reached_refusals_stop_inspection_and_native_consumption() {
    // First native callback controls, shared Controls, then each actual input
    // metadata inspection. Every reached refusal stops the next producer.
    for allowed in 0..5 {
        let (calls,cut,live)=source();let (communication,trace,id)=communication();
        let start=calls.load(Ordering::SeqCst);cut.store(start+allowed,Ordering::SeqCst);
        let error=communication.all_reduce_sum_wave(vec![FakeTensor(vec![1]),FakeTensor(vec![2]),FakeTensor(vec![3])],id,&(),Some(&())).unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst),start+allowed+1);
        assert_eq!(INSPECTIONS.get(),allowed.saturating_sub(2));
        assert_eq!(WAVES.get(),0);assert!(trace.borrow().is_empty());
        drop(communication);SOURCE.with(|source|source.borrow_mut().take());
        if allowed>0 { assert!(live.load(Ordering::SeqCst),"prepared validation failure lost its source"); }
        drop(error);assert!(!live.load(Ordering::SeqCst));
    }
}
#[test]
fn retained_sum_wave_native_failure_preserves_typed_source_and_funding() {
    let (_,_,live)=source();let (communication,trace,id)=communication();FAIL_NATIVE.set(true);
    let error=communication.all_reduce_sum_wave(vec![FakeTensor(vec![1]),FakeTensor(vec![2])],id,&(),Some(&())).unwrap_err();
    assert_eq!(trace.borrow().len(),1);assert_eq!(WAVES.get(),1);
    assert!(matches!(error,PartitionExecutionError::PreparedCommunication{operation:CommunicationOperation::AllReduceSum,..}));
    let mut current: &(dyn std::error::Error+'static)=&error;let mut found=false;
    loop { if current.downcast_ref::<NativeFailure>().is_some(){found=true;break;} match current.source(){Some(source)=>current=source,None=>break} }
    assert!(found,"original typed native leaf was discarded");
    drop(communication);SOURCE.with(|source|source.borrow_mut().take());
    assert!(live.load(Ordering::SeqCst));drop(error);assert!(!live.load(Ordering::SeqCst));
    FAIL_NATIVE.set(false);
}
