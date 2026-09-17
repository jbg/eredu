use super::*;
use crate::backend::runtime::distributed::topology::ParallelCommunicators;
use eredu_core::{Completion as _, CollectiveGroupId, CompletionCancellationMode};
use eredu_runtime::{CommunicationCompletionPolicy, CommunicationGroupDescriptor,
    CommunicationGroupRequirements, CommunicationManifest, CommunicationOperationRequirement,
    PartitionCommunicationAuthority, working_memory::{InferenceExecutionIdentity, WorkingMemoryPool}};
use super::super::prepared::{CompletionResourceLayout,PreparedCompletionResources};

#[derive(Clone,Copy)]
enum Case { Flag(f32),Agreement(i32) }
impl Case {
    fn input(self)->Array { match self { Self::Flag(value)=>Array::from_slice(&[value],&[1]),Self::Agreement(value)=>Array::from_slice(&[value],&[1]) } }
    fn zero(self)->Array { match self { Self::Flag(_)=>Array::from_slice(&[0_f32],&[1]),Self::Agreement(_)=>Array::from_slice(&[0_i32],&[1]) } }
    fn prepare(self,source:&OriginalCommunicationSource<'_>)->PreparedCommunicationScalar {
        match self { Self::Flag(_)=>PreparedCommunicationScalar::prepare_flag(source),Self::Agreement(_)=>PreparedCommunicationScalar::prepare_agreement(source,0) }.unwrap()
    }
    fn ordinary(self,input:&Array)->bool {
        let completion=MlxCommunicationCompletion::submit([input],vec![input.clone()],vec![],vec![],vec![],vec![]).unwrap();
        match self {
            Self::Flag(_)=> { let (result,completion)=completion.with_f32_flag(input.clone());completion.wait().unwrap();result.resolve().unwrap() },
            Self::Agreement(_)=> { let (result,completion)=completion.with_failure_agreement(input.clone(),1);completion.wait().unwrap();result.resolve().unwrap() },
        }
    }
}

#[test]
fn prepared_scalar_readouts_match_ordinary_and_keep_source_after_completion_and_refusal() {
    let stream=Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0));
    let _runtime=safemlx::PrefillRootsRuntime::prepare_for_stream(&stream,&stream).unwrap();
    let world=safemlx::distributed::Group::init(false,safemlx::distributed::Backend::Ring).unwrap();
    let group=CommunicationGroupDescriptor::new(CollectiveGroupId::new(43),0,vec![0],Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)]).unwrap()).unwrap();
    let policy=CommunicationCompletionPolicy::new(std::time::Duration::from_millis(1),
        CompletionCancellationMode::QuarantineUntilComplete).unwrap();
    let manifest=CommunicationManifest::new(1,0,vec![group],vec![]).unwrap().with_completion_policy(policy);
    let actual=ParallelCommunicators::from_manifest(&manifest,&world,&stream).unwrap();
    let authority=PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool=WorkingMemoryPool::new(16<<20,0).unwrap();
    let funding=pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(),16<<20).unwrap();
    let source=actual.bind_original_source(&manifest,&world,&authority,&funding).unwrap();
    let traversal=safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
        roots:1,arrays:5,tape_entries:3,input_edges:5,output_slots:3,streams:1,captures:8,
    }).expect("current native finite Eval traversal");
    let mut escaped=Vec::new();
    let mut refused=None;
    for (index,case) in [Case::Flag(-3.5),Case::Flag(0.0),Case::Flag(f32::NAN),Case::Agreement(1),Case::Agreement(0)].into_iter().enumerate() {
        let input=case.input();let zero=case.zero();input.evaluated().unwrap();zero.evaluated().unwrap();
        let expected=case.ordinary(&input);
        let scalar=case.prepare(&source);
        let layout=CompletionResourceLayout { arrays:2,counts:&[],groups:1,routes:0,streams:1 };
        let mut prepared=PreparedCompletionResources::prepare_original(&source,layout,traversal).unwrap();
        prepared.push_array(input.clone()).unwrap();prepared.push_array(zero.clone()).unwrap();
        prepared.push_group(source.group(0).unwrap().0.clone(),0).unwrap();prepared.push_stream(stream.clone()).unwrap();
        let ready=prepared.finish().unwrap();
        let invalid=if index==0 {
            let ready=PreparedCompletionResources::prepare_original(&source,
                CompletionResourceLayout { arrays:0,counts:&[],groups:0,routes:0,streams:0 },traversal).unwrap().finish().unwrap();
            Some((PreparedCommunicationScalar::prepare_flag(&source).unwrap(),ready,Array::from_slice(&[7_i32],&[1])))
        } else { None };
        let graph=safemlx::PreparedSubmissionGraphQuota::try_new(4<<20,()).unwrap().try_allocate().unwrap();
        let records=safemlx::PreparedSubmissionRecordQuota::try_new(1<<20,()).unwrap().try_allocate().unwrap();
        let failure=safemlx::PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
        let mut scope=safemlx::SubmissionScope::try_begin_retaining(safemlx::PreparedSubmissionScopeOwner::try_new(())
            .unwrap().with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
        scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
        failure.bind_original_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();
        let observer=safemlx::OriginalScopeObserver::require_current().unwrap();
        if let Some((scalar,ready,output))=invalid {
            refused=Some(scalar.submit(ready,&source,&observer,&stream,output).unwrap_err());
        }
        safemlx::OperationEvent::validate_traversal_leaf(&input,&observer).unwrap();
        safemlx::OperationEvent::validate_traversal_leaf(&zero,&observer).unwrap();
        let output=input.add(&zero,&stream).unwrap();
        let (result,completion)=scalar.submit(ready,&source,&observer,&stream,output).unwrap();
        assert!(result.pending(),"submission does not prematurely publish the scalar result");
        completion.wait().unwrap();
        assert!(completion.is_complete().unwrap());
        assert!(!result.pending());
        // A repeated observation reuses the same paid cell and completed backing.
        assert!(completion.is_complete().unwrap());
        drop(completion);scope.seal();drop((input,zero));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            let (outcome,status)=observer.progress().unwrap();
            outcome==safemlx::ScopedSubmissionProgress::Observed && status.is_settled()
        });
        observer.retire_completed_records().unwrap();
        drop((observer,scope,graph,records,failure));
        crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
        escaped.push((result,expected));
    }
    drop(source);drop((actual,authority,funding,stream,world));
    assert!(pool.used_bytes().unwrap()>0);
    for (result,expected) in escaped { assert_eq!(result.resolve().unwrap(),expected); }
    assert!(pool.used_bytes().unwrap()>0,"refused dtype retains its paid source error after scalar receipts retire");
    drop(refused);
    crate::backend::submission_recovery::reap();safemlx::reclaim_allocation_owners();
    assert_eq!(pool.used_bytes().unwrap(),0);
}
