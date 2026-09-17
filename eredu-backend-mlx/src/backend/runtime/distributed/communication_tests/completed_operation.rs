use super::*;
use eredu_runtime::{PartitionCommunicationAuthority,
    working_memory::{InferenceExecutionIdentity,WorkingMemoryPool}};
use safemlx::{OriginalScopeObserver,PreparedInputRuntime,PreparedOriginalBufferBudget,
    PreparedPrefillFailure,PreparedSubmissionGraphQuota,PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner,SubmissionScope};
use crate::backend::runtime::distributed::topology::OriginalRankCollective;

const RANK:&str="EREDU_ACCEPTED_COMMUNICATION_RING_RANK";

#[test]
#[ignore="spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn accepted_ring_operation_completes_after_source_fence_and_retires_custody() {
    super::collectives::run_subgroup_workers(2,
        "backend::runtime::distributed::communication_tests::completed_operation::worker",RANK);
}

#[test]
fn worker() { run_rank(OriginalRankCollective::Sum); }

#[test]
#[ignore="spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn selected_rank_gather_keeps_axis_order_and_accepted_custody_after_fence() {
    super::collectives::run_subgroup_workers(2,
        "backend::runtime::distributed::communication_tests::completed_operation::gather_worker",RANK);
}
#[test]
fn gather_worker() { run_rank(OriginalRankCollective::GatherAxisZero); }

fn run_rank(collective:OriginalRankCollective) {
    let Ok(rank)=std::env::var(RANK) else {return;};
    let rank:usize=rank.parse().unwrap();
    let stream=Stream::new_with_device(&Device::new(DeviceType::Cpu,0));
    let world=distributed::Group::init(true,Backend::Ring).unwrap();
    assert_eq!(world.size(),2);assert_eq!(world.rank(),rank);
    let semantic=match collective {OriginalRankCollective::Sum=>CommunicationOperation::AllReduceSum,
        OriginalRankCollective::GatherAxisZero=>CommunicationOperation::AllGatherEven};
    let requirement=CommunicationOperationRequirement::tensors(semantic,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,2,6,None).unwrap()
            .with_output_tensor_elements(if collective==OriginalRankCollective::Sum {6}else{12}).unwrap(),true).unwrap();
    let descriptor=CommunicationGroupDescriptor::new(CollectiveGroupId::new(83),0,vec![0,1],Some(rank),
        CommunicationGroupRequirements::new([requirement]).unwrap()).unwrap();
    let manifest=CommunicationManifest::new(2,rank,vec![descriptor],vec![]).unwrap()
        .with_completion_policy(completion_policy());
    let actual=std::rc::Rc::new(ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap());
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let _runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&stream,&stream).unwrap();
    let allocator=PreparedInputRuntime::prepare().unwrap();
    let values=[7.0_f32,-11.,23.,31.,-5.,17.].map(|value|value+100.0*rank as f32);
    let base=Array::from_slice(&values,&[2,3]);
    let input=base.transpose_axes(&[1,0],&stream).unwrap();
    input.evaluated().unwrap();
    // The first native eval schedules an unscheduled view; the second enters
    // its existing wait branch and detaches the settled ordinary event before
    // this exact leaf enters a different original completion domain.
    input.evaluated().unwrap();
    let ordinary=match collective {
        OriginalRankCollective::Sum=>distributed::all_sum(&input,&world,&stream),
        OriginalRankCollective::GatherAxisZero=>distributed::all_gather(&input,&world,&stream),
    }.unwrap();
    let expected=ordinary.evaluated().unwrap().try_as_slice::<f32>().unwrap().to_vec();
    match collective {
        OriginalRankCollective::Sum=>assert_eq!(expected,[114.,162.,78.,90.,146.,134.]),
        OriginalRankCollective::GatherAxisZero=>assert_eq!(expected,[7.,31.,-11.,-5.,23.,17.,107.,131.,89.,95.,123.,117.]),
    }
    drop(ordinary);

    let pool=WorkingMemoryPool::new(256<<20,0).unwrap();
    // The same initialized Ring scratch is retained by world, selected group,
    // and repeated wrapper aliases. Physical registration sees one exact key.
    let scratch=world.persistent_storage().unwrap().retain_buffer().unwrap();
    assert!(scratch.bytes()>0);
    let scratch_bytes=scratch.bytes() as u64;
    let other_world=distributed::Group::init(true,Backend::Ring).unwrap();
    let other_scratch=other_world.persistent_storage().unwrap().retain_buffer().unwrap();
    assert_eq!(scratch.identity(),other_scratch.identity());
    assert!(scratch.is_for(&other_world));
    let mut retained=crate::backend::runtime::residency::storage::RetainedStorage::default();
    actual.collect_retained_buffers(&mut retained).unwrap();
    retained.include_group_buffer(Some(&scratch)).unwrap();
    retained.include_group_buffer(Some(&other_scratch)).unwrap();
    assert_eq!(retained.byte_bound().unwrap(),Some(scratch.bytes() as u64));
    let registered_buffers=retained.register(&pool).unwrap();
    assert_eq!(pool.used_bytes().unwrap(),scratch.bytes() as u64);
    drop((scratch,other_scratch,other_world));
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),256<<20).unwrap();
    let standalone=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let before=pool.used_bytes().unwrap();
    let persistent=standalone.world_persistent().unwrap();
    let standalone_cost=pool.used_bytes().unwrap()-before;
    assert!(persistent.native().retained_owner_bytes().unwrap() as u64>scratch_bytes);
    drop(persistent);drop(standalone);
    // Identical native table with no registered keys cannot produce pin credit.
    let empty_pool=WorkingMemoryPool::new(256<<20,0).unwrap();
    let missing=crate::backend::runtime::distributed::topology::OriginalCommunicationOwner::bind(
        &actual,&manifest,&world,&authority,&funding).unwrap()
        .pin_registered_buffers(&empty_pool).err().expect("unregistered buffers cannot receive H credit");
    assert_eq!(empty_pool.used_bytes().unwrap(),0);
    let owner=crate::backend::runtime::distributed::topology::OriginalCommunicationOwner::bind(
        &actual,&manifest,&world,&authority,&funding).unwrap().pin_registered_buffers(&pool).unwrap();
    let source=owner.borrow().unwrap();
    let before=pool.used_bytes().unwrap();
    let persistent=source.world_persistent().unwrap();
    let pinned_cost=pool.used_bytes().unwrap()-before;
    assert_eq!(standalone_cost-pinned_cost,scratch_bytes,
        "only exact registered scratch is excluded from persistent H");
    drop(persistent);
    // The source alias, including escaped failures, now holds the existing-only
    // pin. The original inventory may retire before constructor and completion.
    drop(registered_buffers);
    crate::backend::submission_recovery::reap();
    assert!(pool.used_bytes().unwrap()>=scratch_bytes);
    // Refusal happens before constructor/bank/task creation, while each error
    // retains its own paid source after the original source owner is dropped.
    let unknown=source.prepare_rank_collective(CollectiveGroupId::new(84),&input,collective,&allocator)
        .err().expect("equal geometry does not invent a selected group");
    let other=if collective==OriginalRankCollective::Sum {OriginalRankCollective::GatherAxisZero}else{OriginalRankCollective::Sum};
    let unselected=source.prepare_rank_collective(CollectiveGroupId::new(83),&input,other,&allocator)
        .err().expect("selected group must admit this exact semantic operation");
    let operation=source.prepare_rank_collective(CollectiveGroupId::new(83),&input,collective,&allocator).unwrap();
    let backing=operation.backing_capacity();
    assert!(std::ptr::eq(operation.runtime(),&allocator));
    let graph=PreparedSubmissionGraphQuota::try_new(operation.graph_capacity(),()).unwrap().try_allocate().unwrap();
    let records=PreparedSubmissionRecordQuota::try_new(operation.record_capacity(),()).unwrap().try_allocate().unwrap();
    let budget=PreparedOriginalBufferBudget::try_new(&allocator,backing,()).unwrap().try_allocate().unwrap();
    let failure=PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
    let mut scope=SubmissionScope::try_begin_retaining(PreparedSubmissionScopeOwner::try_new(())
        .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
    scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    let observer=OriginalScopeObserver::require_current().unwrap();
    let accepted=operation.construct_accepted(&source,&observer,&stream).unwrap();
    world.mark_terminal_submission();
    assert!(world.terminal_submission());
    let refused=source.prepare_rank_collective(CollectiveGroupId::new(83),&input,collective,&allocator)
        .err().expect("fenced source cannot construct a new operation");
    drop(source);drop((owner,actual,authority,funding,world));
    assert!(pool.used_bytes().unwrap()>0,"accepted value and destination retain their source/H");
    let (value,completion)=accepted.submit().unwrap();
    completion.wait().unwrap();
    assert!(completion.is_complete().unwrap());
    scope.seal();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        let (outcome,status)=observer.progress().unwrap();
        outcome==safemlx::ScopedSubmissionProgress::Observed && status.is_settled()
    });
    observer.retire_completed_records().unwrap();
    drop((observer,scope,graph,records,budget,failure,stream,allocator));
    // Completion above is the sole submission claim. Leave its original Scope
    // before using the ordinary host-readable wrapper for this completed value;
    // no additional evaluation is requested from a terminal active role.
    assert_eq!(value.value().evaluated().unwrap().try_as_slice::<f32>().unwrap(),expected.as_slice());
    drop((value,completion,input,base));
    crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert!(pool.used_bytes().unwrap()>0,"the independent fresh-source refusal retains its paid error");
    drop((refused,unknown,unselected,missing));
    crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0,"accepted completion and refusal retire without source backedges");
    println!("ACCEPTED_RING_PASS rank={rank} operation={collective:?} values={expected:?}");
}
