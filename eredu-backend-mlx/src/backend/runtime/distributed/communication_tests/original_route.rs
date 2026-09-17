//! The new two-edge completion uses actual local Ring endpoints, not a mock.
use super::*;
use eredu_runtime::{PartitionCommunicationAuthority,
    working_memory::{InferenceExecutionIdentity,WorkingMemoryPool}};
use safemlx::{OriginalScopeObserver,PreparedInputRuntime,PreparedOriginalBufferBudget,
    PreparedPrefillFailure,PreparedSubmissionGraphQuota,PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner,SubmissionScope};
const RANK:&str="EREDU_ORIGINAL_ROUTE_RING_RANK";
#[test]
#[ignore="spawns two local Ring ranks and opens loopback sockets; run explicitly"]
fn original_route_pair_keeps_both_roots_and_completes_after_source_fence() {
    super::collectives::run_subgroup_workers(2,
        "backend::runtime::distributed::communication_tests::original_route::worker",RANK);
}
#[test]
fn worker(){
    let Ok(rank)=std::env::var(RANK) else{return;};
    let rank:usize=rank.parse().unwrap();
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} init");
    let stream=Stream::new_with_device(&Device::new(DeviceType::Cpu,0));
    let world=distributed::Group::init(true,Backend::Ring).unwrap();
    assert_eq!((world.rank(),world.size()),(rank,2));
    let descriptor=CommunicationGroupDescriptor::new(CollectiveGroupId::new(83),0,vec![0,1],Some(rank),
        CommunicationGroupRequirements::new([subgroup_requirement()]).unwrap()).unwrap();
    let route_id=CommunicationRouteId::new(91);
    let manifest=CommunicationManifest::new(2,rank,vec![descriptor],vec![
        CommunicationRouteDescriptor::new(route_id,0,0,1,route_requirement()).unwrap()])
        .unwrap().with_completion_policy(completion_policy());
    let actual=std::rc::Rc::new(ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap());
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} manifest-ready");
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let _runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&stream,&stream).unwrap();
    let allocator=PreparedInputRuntime::prepare().unwrap();
    // 16 MiB per direction exceeds ordinary local socket buffering: making
    // both endpoints send first must not accidentally pass a tiny-message test.
    const ROWS:usize=1024;const COLS:usize=4096;
    let value=|index:usize,rank:usize| (index%97) as f32-48.0+100.0*rank as f32;
    let values=(0..ROWS*COLS).map(|index|value(index,rank)).collect::<Vec<_>>();
    let base=Array::from_slice(&values,&[ROWS as i32,COLS as i32]);
    let input=base.transpose_axes(&[1,0],&stream).unwrap();
    input.evaluated().unwrap();input.evaluated().unwrap();
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} input-settled bytes={}",values.len()*4);
    let pool=WorkingMemoryPool::new(256<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),256<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let exchange=source.route_exchange(0).unwrap().unwrap();
    assert_eq!(exchange.rounds(),1);
    assert_eq!(exchange.route().descriptor().id(),route_id);
    assert!(exchange.source().same_source(source.source()));
    let bad=exchange.round_storage(&source,1,&input).err().expect("no invented second round");
    let round=exchange.round_storage(&source,0,&input).unwrap();
    assert_eq!(round.round(),0);assert!(round.source().same_source(source.source()));
    let prepared=round.prepare(&source,&allocator).unwrap();
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} pair-prepared");
    let graph=PreparedSubmissionGraphQuota::try_new(prepared.graph_capacity(),()).unwrap().try_allocate().unwrap();
    let records=PreparedSubmissionRecordQuota::try_new(prepared.record_capacity(),()).unwrap().try_allocate().unwrap();
    let budget=PreparedOriginalBufferBudget::try_new(&allocator,prepared.backing_capacity(),()).unwrap().try_allocate().unwrap();
    let failure=PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
    let mut scope=SubmissionScope::try_begin_retaining(PreparedSubmissionScopeOwner::try_new(())
        .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
    scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    let observer=OriginalScopeObserver::require_current().unwrap();
    let accepted=prepared.construct_accepted(&source,&observer,&stream).unwrap();
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} pair-accepted");
    world.mark_terminal_submission();
    drop(exchange);drop(source);drop((actual,authority,funding,world));
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} submit");
    let (output,completion)=accepted.submit().unwrap();
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} submitted");
    completion.wait().unwrap();assert!(completion.is_complete().unwrap());
    eprintln!("ORIGINAL_ROUTE_STAGE rank={rank} completed");
    scope.seal();
    crate::backend::submission_recovery::wait_for_retirement(||{
        let (outcome,status)=observer.progress().unwrap();
        outcome==safemlx::ScopedSubmissionProgress::Observed&&status.is_settled()
    });
    observer.retire_completed_records().unwrap();
    drop((observer,scope,graph,records,budget,failure,stream,allocator));
    // Send retains the strided input view; Receive owns row-contiguous bytes.
    // Check both logical tensors without silently compacting the send alias.
    for (which,array) in output.outputs().iter().enumerate() {
        let evaluated=array.evaluated().unwrap();
        let actual=evaluated.try_iter::<f32>().unwrap();
        assert_eq!(actual.len(),ROWS*COLS);
        let expected_rank=if which==0{rank}else{1-rank};
        for (index,actual) in actual.enumerate() {
            let column=index/ROWS;let row=index%ROWS;
            assert_eq!(actual,value(row*COLS+column,expected_rank),"output {which} logical index {index}");
        }
    }
    assert!(pool.used_bytes().unwrap()>0);
    drop((output,completion,input,base));
    crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert!(pool.used_bytes().unwrap()>0,"escaped round refusal retains its source and H");
    drop(bad);crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0);
    println!("ORIGINAL_ROUTE_PAIR_PASS rank={rank} elements={} bytes={} strided=true",ROWS*COLS,ROWS*COLS*4);
}
