use super::*;
use crate::backend::runtime::distributed::topology::ParallelCommunicators;
use eredu_core::{BoundedCompletion as _, CollectiveGroupId, CompletionCancellationMode};
use eredu_runtime::{CommunicationCompletionPolicy, CommunicationGroupDescriptor,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationOperationRequirement,
    PartitionCommunicationAuthority,
    working_memory::{InferenceExecutionIdentity, WorkingMemoryPool}};

#[test]
fn prepared_completion_destinations_keep_actual_sources_refusals_and_quarantined_custody() {
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let world=safemlx::distributed::Group::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let group=CommunicationGroupDescriptor::new(CollectiveGroupId::new(29),0,vec![0],Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let policy=CommunicationCompletionPolicy::new(std::time::Duration::from_millis(1),
        CompletionCancellationMode::QuarantineUntilComplete).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![group],vec![]).unwrap().with_completion_policy(policy);
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(4<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),4<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let empty=CompletionResourceLayout { arrays:0,counts:&[],groups:0,routes:0,streams:0 };
    let other=PreparedCompletionResources::prepare(&source,empty,CompletionResourceKind::Distributed).unwrap().finish().unwrap();
    assert!(matches!(other.orphan,OrphanDestination::Distributed(_)));
    drop(other);
    let incomplete=PreparedCompletionResources::prepare(&source,
        CompletionResourceLayout { arrays:1,..empty },CompletionResourceKind::Distributed).unwrap();
    assert!(incomplete.finish().is_err(),"missing retained roots cannot publish a ready recovery");
    let values=Array::from_slice(&[2_i32,-3,5],&[3]);
    let layout=CompletionResourceLayout { arrays:1,counts:&[3],groups:1,routes:0,streams:1 };
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    let mut prepared=PreparedCompletionResources::prepare(&source,layout,CompletionResourceKind::Communication).unwrap();
    prepared.set_counts(0,&[2,3,5]).unwrap();
    let escaped=prepared.set_counts(0,&[7,11,13]).unwrap_err();
    assert_eq!(prepared.counts[0],[2,3,5],"duplicate writes cannot replace a prepared buffer");
    assert!(prepared.push_group(Group::uncontracted(&world),0).is_err(),
        "equal native world without the retained group contract is not a source");
    prepared.retain_group(0).unwrap();
    prepared.push_array(values.clone()).unwrap();
    prepared.push_stream(stream.clone()).unwrap();
    let ready=prepared.finish().unwrap();
    assert_ne!(ready.recovery.allocation_identity(),0);
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(),0,
        "preparing retention is not collective submission or completion");
    let ReadyCompletionResources { recovery,orphan,consumers:_,custody }=ready;
    drop(recovery); // Never-started native retention uses the existing recovery destructor.
    let OrphanDestination::Communication(destination)=orphan else { panic!("wrong destination") };

    // Exercise the actual existing timeout/reap worker with the prepared node.
    // Event submission here is the ordinary fixture producer, deliberately not
    // a claim that this packet qualifies native distributed event allocation.
    let mut completion=MlxCommunicationCompletion::submit([&values],vec![values.clone()],vec![],
        vec![],vec![],vec![]).unwrap();
    completion.orphan_destination=Some(destination);
    completion.force_pending=true;
    assert!(matches!(completion.wait_bounded(policy.bounded_wait()).unwrap(),
        eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }));
    drop(source);
    drop((actual,authority,funding,custody,values,stream,world));
    assert!(pool.used_bytes().unwrap()>0);
    drop(escaped);
    assert!(pool.used_bytes().unwrap()>0,"the occupied prepared orphan node retains its own funding");
    super::super::communication::release_forced_pending_orphans();
    crate::backend::submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0,"terminal retirement releases node before its funding");
}

#[test]
fn prepared_destination_panicking_predicate_keeps_remaining_owned_queue() {
    use std::{cell::Cell,panic::{catch_unwind,AssertUnwindSafe}};
    struct Value(Rc<Cell<usize>>);
    impl Drop for Value { fn drop(&mut self) { self.0.set(self.0.get()+1); } }
    let drops=Rc::new(Cell::new(0));
    let mut queue=destinations::Destinations::new();
    for _ in 0..3 { queue.push_prepared(Value(drops.clone()),Destination::new(None)); }
    assert!(catch_unwind(AssertUnwindSafe(||queue.retain(|_|panic!("predicate")))).is_err());
    assert_eq!(queue.len(),3); assert_eq!(drops.get(),0);
    queue.clear(); assert_eq!(drops.get(),3);
}

#[test]
fn original_communication_event_completes_inside_current_role_and_keeps_prepaid_quarantine() {
    use eredu_core::Completion as _;
    use safemlx::{OperationEvent, OperationEvalTraversalLimits, OriginalScopeObserver,
        PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
        PreparedSubmissionScopeOwner, SubmissionScope};
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let _runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&stream,&stream).unwrap();
    let world=safemlx::distributed::Group::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let group=CommunicationGroupDescriptor::new(CollectiveGroupId::new(37),0,vec![0],Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let policy=CommunicationCompletionPolicy::new(std::time::Duration::from_millis(1),
        CompletionCancellationMode::QuarantineUntilComplete).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![group],vec![]).unwrap().with_completion_policy(policy);
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(8<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),8<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let traversal=OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots:1,arrays:5,tape_entries:3,input_edges:5,output_slots:3,streams:1,captures:8,
    }).expect("current native prepared Eval producer");
    let left=Array::from_slice(&[2.0f32,3.,5.],&[3]);
    let right=Array::from_slice(&[7.0f32,11.,13.],&[3]);
    left.evaluated().unwrap(); right.evaluated().unwrap();
    let layout=CompletionResourceLayout { arrays:2,counts:&[],groups:1,routes:0,streams:1 };
    let mut prepared=PreparedCompletionResources::prepare_original(&source,layout,traversal).unwrap();
    prepared.push_array(left.clone()).unwrap(); prepared.push_array(right.clone()).unwrap();
    prepared.retain_group(0).unwrap();
    prepared.push_stream(stream.clone()).unwrap();
    let ready=prepared.finish().unwrap();
    let bad=PreparedCompletionResources::prepare_original(&source,
        CompletionResourceLayout { arrays:0,counts:&[],groups:0,routes:0,streams:0 },traversal).unwrap().finish().unwrap();

    // The enclosing admitted-role fixture owns actual native Graph/Record/failure
    // resources. The completion producer neither allocates nor substitutes them.
    let graph=PreparedSubmissionGraphQuota::try_new(4<<20,()).unwrap().try_allocate().unwrap();
    let records=PreparedSubmissionRecordQuota::try_new(1<<20,()).unwrap().try_allocate().unwrap();
    let failure=PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
    let mut scope=SubmissionScope::try_begin_retaining(PreparedSubmissionScopeOwner::try_new(())
        .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
    scope.enable_scoped_observation().unwrap(); scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap(); scope.enable_original_native_controls().unwrap();
    let observer=OriginalScopeObserver::require_current().unwrap();
    let refused=bad.submit_original(&source,&observer,&stream,&[]).unwrap_err();
    OperationEvent::validate_traversal_leaf(&left,&observer).unwrap();
    OperationEvent::validate_traversal_leaf(&right,&observer).unwrap();
    let sum=left.add(&right,&stream).unwrap();
    let value=sum.multiply(&left,&stream).unwrap();
    let mut completion=ready.submit_original(&source,&observer,&stream,std::slice::from_ref(&value)).unwrap();
    completion.wait().unwrap(); // Scope is still active: this must not wait for its seal.
    assert!(completion.is_complete().unwrap());
    assert!(completion.0.event.original_observer().unwrap().same_scope(&observer));
    assert_eq!(value.evaluated().unwrap().try_as_slice::<f32>().unwrap(),&[18.,42.,90.]);
    completion.0.force_pending=true;
    assert!(matches!(completion.wait_bounded(policy.bounded_wait()).unwrap(),
        eredu_core::BoundedCompletionOutcome::DeadlineExceeded { .. }));
    scope.seal();
    drop(source); drop((actual,authority,funding));
    assert!(pool.used_bytes().unwrap()>0);
    drop(refused);
    assert!(pool.used_bytes().unwrap()>0,"prepared quarantine keeps its independent source/account");
    super::super::communication::release_forced_pending_orphans();
    drop((sum,value,left,right));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        let (outcome,status)=observer.progress().unwrap();
        outcome==safemlx::ScopedSubmissionProgress::Observed && status.is_settled()
    });
    observer.retire_completed_records().unwrap();
    drop((observer,scope,graph,records,failure,stream,world));
    crate::backend::submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0);
}

#[test]
fn selected_completion_copies_keep_source_order_and_failure_custody_without_submission() {
    use eredu_core::checkpoint::TensorDtype;
    use eredu_runtime::{CommunicationOperation,CommunicationTensorLimits};
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let world=safemlx::distributed::Group::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let requirements=CommunicationGroupRequirements::new([
        CommunicationOperationRequirement::barrier(true),
        CommunicationOperationRequirement::tensors(CommunicationOperation::AllReduceSum,
            [TensorDtype::F32],CommunicationTensorLimits::new(1,3,4097,None).unwrap(),true).unwrap(),
    ]).unwrap();
    let group=CommunicationGroupDescriptor::new(CollectiveGroupId::new(47),0,vec![0],Some(0),requirements).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![group],vec![]).unwrap().with_completion_policy(
        CommunicationCompletionPolicy::new(std::time::Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete).unwrap());
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(4<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),4<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let mut prepared=PreparedCompletionResources::prepare(&source,
        CompletionResourceLayout{arrays:0,counts:&[],groups:2,routes:0,streams:0},
        CompletionResourceKind::Communication).unwrap();
    crate::backend::runtime::distributed::group::reset_native_collective_submissions();
    prepared.retain_control_world().unwrap();
    let first=prepared.retain_group(1).unwrap_err();
    assert_eq!(prepared.groups.len(),1,"invalid source lookup leaves the paid prefix unchanged");
    prepared.retain_group(0).unwrap();
    let second=prepared.retain_group(0).unwrap_err();
    assert_eq!(prepared.groups.len(),2,"full destination cannot be replaced or overfilled");
    assert!(source.matches_world(&prepared.groups[0]));
    assert!(source.matches_group(0,&prepared.groups[1]));
    assert!(prepared.groups[0].shares_native_world(&prepared.groups[1]));
    assert_eq!(crate::backend::runtime::distributed::group::native_collective_submissions(),0);
    let ready=prepared.finish().unwrap();
    drop(source);drop((actual,authority,funding,world,stream));
    assert!(pool.used_bytes().unwrap()>0);
    drop(ready);
    assert!(pool.used_bytes().unwrap()>0,"escaped invalid selections keep independently funded errors");
    drop(first);
    assert!(pool.used_bytes().unwrap()>0);
    drop(second);
    crate::backend::submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0);
}
