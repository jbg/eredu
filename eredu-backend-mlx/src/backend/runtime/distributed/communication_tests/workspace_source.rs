//! Actual two-rank source queries; no standalone completion of a lazy model partial.
use super::*;
use eredu_runtime::{PartitionCommunicationAuthority,working_memory::{InferenceExecutionIdentity,WorkingMemoryPool}};
use eredu_nn::workspace::{WorkspaceLayout,WorkspaceDtype,WorkspaceRepresentation,WorkspaceFloatingType,
    WorkspaceOperationView,WorkspaceOperationKindView,WorkspaceCollectiveView,WorkspaceLayoutList};
const RANK:&str="EREDU_WORKSPACE_COMMUNICATION_RING_RANK";
#[test]
#[ignore="opens two local Ring ranks; run explicitly"]
fn workspace_sum_source_binds_lazy_partials_and_retains_refused_sources() {
    super::collectives::run_subgroup_workers(2,
        "backend::runtime::distributed::communication_tests::workspace_source::worker",RANK);
}
#[test]
fn worker() {
    let Ok(rank)=std::env::var(RANK) else{return;};let rank:usize=rank.parse().unwrap();
    let stream=Stream::new_with_device(&Device::new(DeviceType::Cpu,0));
    let world=distributed::Group::init(true,Backend::Ring).unwrap();
    assert_eq!((world.size(),world.rank()),(2,rank));
    let id=CollectiveGroupId::new(93);
    let requirement=CommunicationOperationRequirement::tensors(CommunicationOperation::AllReduceSum,
        [TensorDtype::F32],CommunicationTensorLimits::new(1,2,6,None).unwrap(),true).unwrap();
    let descriptor=CommunicationGroupDescriptor::new(id,0,vec![0,1],Some(rank),
        CommunicationGroupRequirements::new([requirement]).unwrap()).unwrap();
    let manifest=CommunicationManifest::new(2,rank,vec![descriptor],vec![]).unwrap().with_completion_policy(completion_policy());
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(256<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),256<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let layout=[WorkspaceLayout::new(&[2,3],WorkspaceDtype::Float32).unwrap().with_representation(
        Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32,false)))];
    let equation=WorkspaceOperationView{kind:WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum{partitions:2,rank}),
        inputs:WorkspaceLayoutList::Owned(&layout),outputs:WorkspaceLayoutList::Owned(&layout)};
    let quote=source.sum_workspace_storage(id,equation).unwrap();
    assert_eq!(quote.native().graph_population(),(1,1));assert_eq!(quote.native().output_geometry(),(2,6));
    assert!(quote.native().graph_extent().unwrap()>0);
    let base=Array::from_slice(&[7.0_f32,-11.,23.,31.,-5.,17.],&[2,3]);
    let lazy=base.add(&base,&stream).unwrap();
    let bound=quote.bind_actual(&source,&lazy).unwrap();
    assert_eq!(bound.native().constructor().output_geometry(),(2,6));
    assert!(bound.with_completion().is_err(),"layout binding must not settle the lazy predecessor to qualify standalone completion");
    let wrong=Array::from_slice(&[7_i32,-11,23,31,-5,17],&[2,3]);
    let mismatch=source.sum_workspace_storage(id,equation).unwrap().bind_actual(&source,&wrong)
        .err().expect("same shape cannot substitute a different physical dtype");
    let logical=[WorkspaceLayout::new(&[2,3],WorkspaceDtype::Float32).unwrap()];
    let ambiguous=WorkspaceOperationView{inputs:WorkspaceLayoutList::Owned(&logical),outputs:WorkspaceLayoutList::Owned(&logical),..equation};
    let missing=source.sum_workspace_storage(id,ambiguous).err().expect("logical F32 cannot invent actual floating representation");
    let mut cause:Option<&(dyn std::error::Error+'static)>=Some(&missing);
    let diagnostic=loop {
        let current=cause.expect("the retained source error must expose its fixed scalar cause");
        let text=current.to_string();
        if text.contains("lacks floating scalar evidence") {break text;}
        cause=current.source();
    };
    assert!(diagnostic.contains("AllReduceSum"));
    assert!(diagnostic.contains(&format!("local rank {rank}/2")));
    assert!(diagnostic.contains("input rank 2, width Some(3), elements Some(6)"));
    let changed=WorkspaceOperationView{kind:WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum{partitions:2,rank:1-rank}),..equation};
    let foreign=source.sum_workspace_storage(id,changed).err().expect("equal geometry must retain exact local rank");
    drop(source);drop((actual,authority,funding,world,lazy,base,wrong));
    assert!(pool.used_bytes().unwrap()>0,"escaped per-query failures retain their exact source/H");
    drop((mismatch,missing,foreign));
    assert_eq!(pool.used_bytes().unwrap(),0);
    println!("WORKSPACE_RING_SOURCE_PASS rank={rank}");
}
