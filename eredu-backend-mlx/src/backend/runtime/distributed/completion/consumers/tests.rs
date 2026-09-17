use super::*;
use crate::backend::runtime::distributed::topology::ParallelCommunicators;
use eredu_core::{Completion as _,CollectiveGroupId,CompletionCancellationMode};
use eredu_runtime::{CommunicationCompletionPolicy,CommunicationGroupDescriptor,CommunicationGroupRequirements,
    CommunicationManifest,CommunicationOperationRequirement,PartitionCommunicationAuthority,
    working_memory::{InferenceExecutionIdentity,WorkingMemoryPool}};
use super::super::prepared::{PreparedCompletionResources,CompletionResourceLayout};

#[test]
fn original_stream_consumers_spend_exact_slots_and_retain_resources_after_primary_drop() {
    let device=safemlx::Device::new(safemlx::DeviceType::Cpu,0);
    let producer=Stream::new_with_device(&device);let first=Stream::new_with_device(&device);
    let second=Stream::new_with_device(&device);let foreign=Stream::new_with_device(&device);
    let _runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&producer,&producer).unwrap();
    // Exact consumer workers belong to cold native setup, before the original
    // role. A retained Stream handle alone does not initialize its worker.
    let _consumer_runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&first,&second).unwrap();
    let world=safemlx::distributed::Group::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let group=CommunicationGroupDescriptor::new(CollectiveGroupId::new(53),0,vec![0],Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let policy=CommunicationCompletionPolicy::new(std::time::Duration::from_millis(1),CompletionCancellationMode::QuarantineUntilComplete).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![group],vec![]).unwrap().with_completion_policy(policy);
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&producer).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(8<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),8<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let left=Array::from_slice(&[2.0f32,3.,5.],&[3]);let right=Array::from_slice(&[7.0f32,11.,13.],&[3]);
    left.evaluated().unwrap();right.evaluated().unwrap();
    let traversal=safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
        roots:1,arrays:5,tape_entries:3,input_edges:5,output_slots:3,streams:1,captures:8,
    }).expect("current finite native Eval traversal");
    let waits=safemlx::OperationEvent::wait_record_layout(2).expect("current finite native WaitRecord producer");
    let mut prepared=PreparedCompletionResources::prepare_original_with_consumers(&source,
        CompletionResourceLayout { arrays:2,counts:&[],groups:1,routes:0,streams:3 },traversal,waits).unwrap();
    prepared.push_array(left.clone()).unwrap();prepared.push_array(right.clone()).unwrap();
    prepared.push_group(source.group(0).unwrap().0.clone(),0).unwrap();
    for stream in [&producer,&first,&second] { prepared.push_stream(stream.clone()).unwrap(); }
    let ready=prepared.finish().unwrap();
    let graph=safemlx::PreparedSubmissionGraphQuota::try_new(4<<20,()).unwrap().try_allocate().unwrap();
    let records=safemlx::PreparedSubmissionRecordQuota::try_new(1<<20,()).unwrap().try_allocate().unwrap();
    let failure=safemlx::PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
    let mut scope=safemlx::SubmissionScope::try_begin_retaining(safemlx::PreparedSubmissionScopeOwner::try_new(())
        .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
    scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();
    let observer=OriginalScopeObserver::require_current().unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&left,&observer).unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&right,&observer).unwrap();
    let value=left.add(&right,&producer).unwrap();
    let completion=ready.submit_original(&source,&observer,&producer,std::slice::from_ref(&value)).unwrap();
    let consumers=completion.0.consumers.as_ref().unwrap();
    let buffer=consumers.slots.borrow().active.as_ptr();
    assert!(completion.wait_on(&foreign,&observer).is_err());
    assert_eq!(consumers.slots.borrow().ready.remaining(),2,"foreign stream never consumes the caller's admitted wait slot");
    completion.wait_on(&first,&observer).unwrap();completion.wait_on(&second,&observer).unwrap();
    assert!(completion.wait_on(&first,&observer).is_err(),"accepted waits cannot refill their finite bank");
    assert_eq!(consumers.slots.borrow().ready.remaining(),0);
    assert_eq!(consumers.slots.borrow().active.as_ptr(),buffer);
    let a=value.multiply(&left,&first).unwrap();let b=value.add(&right,&second).unwrap();
    let outputs=[a,b];
    let final_plan=safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
        roots:2,arrays:8,tape_entries:5,input_edges:10,output_slots:5,streams:3,captures:16,
    }).unwrap();
    let final_event=safemlx::transforms::async_eval_with_original_prepared_traversal(&outputs,&observer,&producer,&final_plan).unwrap();
    // Primary disposal must not release dependencies of either accepted wait.
    drop(completion);drop(source);drop((actual,authority,funding));
    assert!(pool.used_bytes().unwrap()>0);
    final_event.synchronize().unwrap();
    assert_eq!(outputs[0].completed_in_original_scope(&observer).unwrap().try_as_slice::<f32>().unwrap(),&[18.,42.,90.]);
    assert_eq!(outputs[1].completed_in_original_scope(&observer).unwrap().try_as_slice::<f32>().unwrap(),&[16.,25.,31.]);
    scope.seal();drop((final_event,outputs,value,left,right));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        let (outcome,status)=observer.progress().unwrap();
        outcome==safemlx::ScopedSubmissionProgress::Observed && status.is_settled()
    });
    observer.retire_completed_records().unwrap();drop((observer,scope,graph,records,failure));
    crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0);
}
