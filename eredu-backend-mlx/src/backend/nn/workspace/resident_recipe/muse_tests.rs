//! Released Muse assistant-only fused invocation over actual retained parameters.
//! The target configuration proves compatibility; no target payload is loaded.
use super::*;
use crate::{MlxTensor, backend::{MlxAcceleratorFamily,MlxBackend,MlxDeviceIdentity,
    managed_memory::gpu_stream::PreparedExecutionStreams,nn::shared::MlxNeuralBackend,
    runtime::{cache::kv::ConcatKeyValueCache,checkpoint::binding::populate_module_from_arrays_excluding}},
    composition::mlx::speculative::{MlxAssistantPreparationVisitor,MlxExternalAssistant}};
use eredu_architectures::{ExternalAssistantArchitecture,ExternalAssistantTargetProfile,
    MaterializedExternalAssistantVisitor,external_assistant::invocation::ExternalAssistantOperation,
    muse_glimmer::{DecoderConfig,assistant::{DFlash,DFlashConfig,invocation::{RawContext,
        RawContextArguments,UpdateContext,UpdateContextArguments,FusedProposal,FusedProposalArguments}}}};
use eredu_nn::{ParameterMetadata,ParameterVisitorMut,Parameterized,Tensor};
use safemlx::{Array,DeviceType,Dtype,OperationEvent,OriginalBufferBudget,OriginalScopeObserver,
    PrefillRoots,PrefillRootsRuntime,PreparedOriginalBufferBudget,PreparedPrefillFailure,
    PreparedSubmissionGraphQuota,PreparedSubmissionRecordQuota,PreparedSubmissionScopeOwner,PreparedPipelineCachePlan,
    Stream,SubmissionScope};
use std::{collections::BTreeMap,sync::{Arc,atomic::{AtomicBool,Ordering}}};

fn tensor(shape:&[i32],phase:usize)->MlxTensor {
    let count:usize=shape.iter().map(|n|usize::try_from(*n).unwrap()).product();
    let data=(0..count).map(|i|half::bf16::from_f32((((i*13+phase*7)%31) as f32-15.0)*0.025)).collect::<Vec<_>>();
    MlxTensor::from_array(Array::from_slice(&data,shape))
}
fn values(value:&Array)->Vec<f32> {
    let read=value.evaluated().unwrap();
    match value.dtype() {
        Dtype::Bfloat16=>read.try_to_vec::<half::bf16>().unwrap().into_iter().map(f32::from).collect(),
        Dtype::Float32=>read.try_to_vec::<f32>().unwrap(),
        other=>panic!("unexpected released floating result {other:?}"),
    }
}
fn compare(actual:&Array,expected:&[f32]) {
    let actual=values(actual);assert_eq!(actual.len(),expected.len());
    for(a,b)in actual.into_iter().zip(expected){
        assert!(a.is_finite()&&(a-b).abs()<=0.02+0.02*b.abs(),"actual {a}, ordinary {b}");
    }
}
fn project(value:&Array,context:&WorkspaceContext)->WorkspaceTensor {
    let descriptor=value.try_descriptor().unwrap();
    let floating=match descriptor.facts().dtype(){Dtype::Bfloat16=>WorkspaceFloatingType::Bfloat16,
        Dtype::Float32=>WorkspaceFloatingType::Float32,other=>panic!("unexpected released source dtype {other:?}")};
    assert!(descriptor.facts().allocation().is_some());
    WorkspaceTensor::existing(context.layout(descriptor.shape(),WorkspaceDtype::Float32).unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(floating,descriptor.row_contiguous().unwrap()))),context).unwrap()
}
struct BindQuote<'a>{sources:&'a BTreeMap<String,Array>,count:usize,context:&'a WorkspaceContext}
impl<'a> ParameterVisitorMut<'a,WorkspaceTensor> for BindQuote<'_>{
    fn visit_mut(&mut self,metadata:eredu_nn::ParameterMetadataView<'_>,value:&'a mut WorkspaceTensor){
        let source=self.sources.get(metadata.id().as_str()).expect("actual retained parameter binding");
        assert_eq!(value.shape(),source.shape());*value=project(source,self.context);self.count+=1;
    }
}
#[derive(Debug)]struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime{fn drop(&mut self){self.0.store(true,Ordering::SeqCst);}}
struct Invoke<'a>{config:&'a DFlashConfig,stream:&'a Stream,allocator:&'a safemlx::PreparedInputRuntime}
impl MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor> for Invoke<'_>{
    type Output=();
    fn visit<A:ExternalAssistantArchitecture>(self,assistant:&mut MlxExternalAssistant<A>){
        assert_eq!(A::configuration_model_type(&assistant.config),"muse_glimmer_assistant");
        assert!(A::quantization(&assistant.config).is_none());
        // Rebind the same retained arrays into the typed test equation. There
        // is no second materialization or independent parameter backing.
        let mut module=DFlash::<MlxNeuralBackend>::new(self.config.clone(),self.stream).unwrap();
        populate_module_from_arrays_excluding(&mut module,assistant.source.values(),|_|false).unwrap();
        let sources=assistant.source.values();assert!(!sources.is_empty());
        let taps=(0..self.config.target_layer_ids.len()).map(|i|tensor(&[1,2,self.config.hidden_size],101+i)).collect::<Vec<_>>();
        let embeddings=tensor(&[1,self.config.block_size as i32,self.config.hidden_size],211);
        let raw=RawContext::execute::<MlxNeuralBackend,ConcatKeyValueCache>(&mut module,
            RawContextArguments{previous:None,states:&taps},self.stream).unwrap();
        let committed=UpdateContext::execute::<MlxNeuralBackend,ConcatKeyValueCache>(&mut module,
            UpdateContextArguments{previous:None,pending:&raw,absolute_end:2},self.stream).unwrap();
        assert_eq!((committed.start,committed.end),(0,2));
        assert_eq!(committed.layers.len(),self.config.num_hidden_layers as usize);
        let arguments=FusedProposalArguments{embeddings:&embeddings,committed:&committed,absolute_end:2};
        FusedProposal::visit_arguments(&arguments,&mut|v|{v.as_array().evaluated().unwrap();});
        let reference=FusedProposal::execute::<MlxNeuralBackend,ConcatKeyValueCache>(&mut module,
            FusedProposalArguments{embeddings:&embeddings,committed:&committed,absolute_end:2},self.stream).unwrap();
        let expected=values(reference.as_array());
        assert!(expected.iter().all(|v|v.is_finite())&&expected.iter().any(|v|v.abs()>1e-5));drop(reference);
        // The first ordinary eval schedules and waits for each lazy root, but
        // its Array descriptor can still carry the completed ordinary event.
        // Cross the existing wait/available boundary for every exact borrowed
        // argument before projecting or lending it to a new original Scope.
        FusedProposal::visit_arguments(&arguments,&mut|v|{v.as_array().evaluated().unwrap();});
        let mechanism=MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context=WorkspaceContext::new(mechanism);
        let mut quoted=FusedProposal::workspace_module(self.config,&context).unwrap();
        let mut binding=BindQuote{sources,count:0,context:&context};quoted.visit_parameters_mut(&mut binding);
        assert_eq!(binding.count,sources.len());
        let mut projected=FusedProposal::project(&arguments,|v|Ok(project(v.as_array(),&context)),&context).unwrap();
        let geometry=FusedProposal::geometry(&projected,&context).unwrap();
        let mut opening=Vec::new();FusedProposal::visit_projected(&projected,&mut|v|opening.push(v.clone()));
        context.begin_state_span(&opening).unwrap();
        let traced=FusedProposal::trace(&mut quoted,&mut projected,&context).unwrap();
        let retained=opening.len();let mut closing=Vec::new();
        FusedProposal::visit_projected(&projected,&mut|v|closing.push(v.clone()));closing.push(traced);
        let report=context.report(&closing).unwrap();
        let recorder=ResidentRecipeRecorder::with_context(geometry,mechanism,&context).unwrap();
        let reduced=recorder.reduce_trace(&report,None,retained,1).unwrap();
        assert_eq!(reduced.first_missing_operation,None,"released fused source must be complete");
        assert_eq!(reduced.validation_roots,0);
        assert_eq!(reduced.nested_completions,0,"short fused block has no explicit tiled completion");
        let storage=reduced.mutable_storage.unwrap();
        let completion=ResidentCompletionRecipe{validation_roots:0,grouped_outputs:reduced.grouped_outputs,
            traversal:reduced.traversal.unwrap(),graph:reduced.graph.unwrap(),dispatch:reduced.dispatch,
            nested_completions:0,nested_root_capacity:0};
        assert_eq!(completion.traversal.roots(),closing.len());
        let graph_bytes=usize::try_from(graph_capacity::ResidentGraphStorage::for_completion(completion).unwrap().full_capacity.unwrap()).unwrap();
        let record_bytes=usize::try_from(record_capacity::ResidentRecordStorage::for_completion(completion).unwrap().full_capacity.unwrap()).unwrap();
        let physical=OriginalBufferBudget::metal_population_layout(self.allocator,
            usize::try_from(storage.mutable_bytes()).unwrap(),storage.maximum_births()).unwrap().capacity();
        assert!(graph_bytes>0&&record_bytes>0&&physical>0);
        // Independent sentinels distinguish a retained native domain from a
        // delayed final Rust owner. They grant no storage or completion credit.
        let released: [Arc<AtomicBool>;6] = std::array::from_fn(|_| Arc::new(AtomicBool::new(false)));
        let owners = released.each_ref().map(|flag| Arc::new(Lifetime(flag.clone())));
        let retired = || released.iter().all(|flag| flag.load(Ordering::SeqCst));
        let graph=PreparedSubmissionGraphQuota::try_new(graph_bytes,owners[0].clone()).unwrap().try_allocate().unwrap();
        // GPU selectors need the same finite request-owned destination as the
        // production original_domains worker, even after the ordinary oracle
        // warmed its independent platform cache. Use the actual recipe count.
        let pipeline_plan=PreparedPipelineCachePlan::new(completion.dispatch.expect("Metal dispatch source").kernel_attempts);
        context.charge_metadata(pipeline_plan.layout::<Arc<Lifetime>>().unwrap().required_bytes().unwrap()).unwrap();
        let pipeline=pipeline_plan.realize(owners[5].clone()).unwrap();pipeline.install(&graph).unwrap();
        let records=PreparedSubmissionRecordQuota::try_new(record_bytes,owners[1].clone()).unwrap().try_allocate().unwrap();
        let budget=PreparedOriginalBufferBudget::try_new(self.allocator,physical,owners[2].clone()).unwrap().try_allocate().unwrap();
        let failure=PreparedPrefillFailure::try_new(owners[3].clone()).unwrap().try_allocate().unwrap();
        let runtime=PrefillRootsRuntime::prepare_for_stream(self.stream,self.stream).unwrap();
        let mut roots=PrefillRoots::new_retained(&runtime,closing.len(),&graph,&failure).unwrap();
        let mut scope=SubmissionScope::try_begin_retaining(PreparedSubmissionScopeOwner::try_new(owners[4].clone()).unwrap()
            .with_graph_quota(graph.clone()).with_record_quota(records.clone())).unwrap();
        scope.enable_scoped_observation().unwrap();scope.require_original_native_controls().unwrap();
        roots.bind_scope(&scope).unwrap();scope.enable_original_native_controls().unwrap();scope.bind_original_buffer_budget(&budget).unwrap();
        let observer=OriginalScopeObserver::require_current().unwrap();
        for value in sources.values(){OperationEvent::validate_traversal_leaf(value,&observer).unwrap();}
        FusedProposal::visit_arguments(&arguments,&mut|v|OperationEvent::validate_traversal_leaf(v.as_array(),&observer).unwrap());
        let bank=OperationEvent::prepare_resident_graph(completion.graph,&observer).unwrap();
        let output=FusedProposal::execute_with_metadata::<MlxNeuralBackend,ConcatKeyValueCache>(&mut module,
            FusedProposalArguments{embeddings:&embeddings,committed:&committed,absolute_end:2},self.stream,&context).unwrap();
        drop(bank);
        FusedProposal::visit_arguments(&arguments,&mut|v|roots.append(v.as_array()).unwrap());roots.append(output.as_array()).unwrap();
        roots.complete_current_scope_on_stream_prepared(self.stream,&completion.traversal)
            .unwrap_or_else(|cause|panic!("complete released Muse fused roots: {cause}"));
        assert!(!observer.status().failed());assert!(budget.occupied_bytes()>0&&budget.occupied_bytes()<=physical);
        compare(output.as_array(),&expected);
        // A ready result event is distinct from the observer's last Record
        // scan: completion may advance between progress_scoped and observe.
        // Retain this exact observer until every accepted Record is settled,
        // then use its owned retirement path before dropping the source. The
        // two observation/retirement loops share the original ten-second cap.
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(10);
        loop {
            let (progress,status)=observer.progress().unwrap();
            assert!(!status.failed()&&!status.blocked(),"released Muse terminal observation: {status:?}");
            if progress==safemlx::ScopedSubmissionProgress::Observed && status.is_settled(){break;}
            assert!(std::time::Instant::now()<deadline,"released Muse terminal observation: {progress:?}, {status:?}");
            std::thread::yield_now();
        }
        assert_eq!(observer.retire_completed_records().unwrap(),safemlx::SubmissionRetirement::CompleteSnapshot);
        scope.seal();
        safemlx::try_with_submission_retirement(||drop((roots,scope,observer,failure,records,graph,budget,pipeline))).unwrap();
        drop((owners,module,committed,raw,taps,embeddings));safemlx::reclaim_allocation_owners();
        assert!(!retired(),"escaped fused output retains actual native custody");
        compare(output.as_array(),&expected);drop(output);
        loop {
            safemlx::try_retire_completed_submissions().unwrap();
            MlxNeuralBackend::reclaim_retired_resources();safemlx::reclaim_allocation_owners();
            if retired(){break;}
            let state=released.each_ref().map(|flag|flag.load(Ordering::SeqCst));
            assert!(std::time::Instant::now()<deadline,
                "released Muse native domains retired [Graph, Record, Buffer, failure, Scope, pipeline]: {state:?}");
            std::thread::yield_now();
        }
        // The materialized assistant retains actual checkpoint/source registration
        // until this visitor has completed and the result backing has retired.
    }
}
#[test]
#[ignore="requires pinned released Muse assistant artifact and qualified Metal execution"]
fn released_muse_fused_source_matches_ordinary_and_retains_escaped_output(){
    let path=std::path::PathBuf::from(std::env::var_os("EREDU_RELEASED_MUSE_ASSISTANT").expect("released assistant path"));
    let target_path=std::env::var_os("EREDU_RELEASED_MUSE_TARGET_CONFIG").expect("released target config path");
    let config=DFlashConfig::from_hf_json(&std::fs::read(path.join("config.json")).unwrap()).unwrap();config.validate_released().unwrap();
    let target=ExternalAssistantTargetProfile::MuseGlimmer(DecoderConfig::from_hf_json(&std::fs::read(target_path).unwrap()).unwrap());
    let selected=eredu_architectures::prepare_external_assistant(&path).unwrap()
        .select_materialization(None,eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS,
            MlxAssistantPreparationVisitor::lowering_for_native_source_test).unwrap()
        .prove_target_compatibility(&target).unwrap();
    let prepared=selected.prepare_source(eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS).unwrap();
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let streams=PreparedExecutionStreams::for_factory(&pool).unwrap().unwrap();
    // This fixture requires the qualified Metal factory. Retain the identity
    // of its actual execution device, including the concrete accelerator family.
    let device=streams.execution().get_device().unwrap();
    assert_eq!(device.get_type().unwrap(),DeviceType::Gpu);
    let identity=MlxDeviceIdentity::from_realized_device(&device,Some(MlxAcceleratorFamily::Metal)).unwrap();
    let backend=MlxBackend::for_prepared_execution_plan(streams,identity);
    backend.validate_original_stream_owners().unwrap();
    let environment=backend.original_copy_environment().unwrap();let allocator=environment.input_runtime().unwrap();
    let mut materialized=prepared.visit(MlxAssistantPreparationVisitor::for_native_source_test(environment.stream(),&pool)).unwrap();
    materialized.visit(Invoke{config:&config,stream:environment.stream(),allocator:&allocator});
}
