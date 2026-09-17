//! Actual original CPU numerical capture with shared raw/mask workers.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::{DefaultSampler,SpeculativeSampler};
use eredu_runtime::input::host::{HostInputPart,HostTensorValues,HostTensorView,PreparedHostInputPlan};
use eredu_core::{capture::*,ObservationPoint,ObservationValueType,ObservationDtype,TensorAxis,
    SymbolicDimension,ObservationRequirement,ObservationPosition,ObservationCatalog,
    DescriptionCompleteness,ObservationSupportReport,ObservationSupport,ObservationSupportStatus,
    TensorObservationData};

fn declaration(transform:Option<CaptureTransform>,vocabulary:usize)->AdmittedCapturePlan{
    declaration_with_attempts(transform,vocabulary,1)
}
fn declaration_with_attempts(transform:Option<CaptureTransform>,vocabulary:usize,attempts:u64)->AdmittedCapturePlan{
    assert!(attempts>0);
    let path=eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
    let point=ObservationPoint{path:path.into(),node_id:"actual.cpu.logits".into(),meaning:"raw model output".into(),
        value_type:ObservationValueType::Tensor,dtype:ObservationDtype::Floating,
        // Use the same symbolic axes as the model-logits discovery contract.
        // Terminal-row reducers resolve Batch/Sequence from this actual request.
        axes:Some([("batch",SymbolicDimension::Batch),("sequence",SymbolicDimension::Sequence),
            ("vocabulary",SymbolicDimension::Known(vocabulary))].into_iter().map(|(name,dimension)|TensorAxis{
            name:name.into(),dimension}).collect()),
        prefill:true,decode:true,requirements:vec![ObservationRequirement::ActivationHooks],
        position:ObservationPosition::BeforeIntervention,retained_bytes:None,host_bytes:None};
    let capabilities=CaptureCapabilities{transformations:vec![CaptureTransformKind::FullTensor,
        CaptureTransformKind::Slice,CaptureTransformKind::Preview,CaptureTransformKind::Summary,CaptureTransformKind::TokenScores,CaptureTransformKind::TopCandidates,CaptureTransformKind::Histogram],max_histogram_bins:3,
        physical_native_limit:false,conditions:vec![]};
    let support=ObservationSupportReport{schema_version:1,capture:capabilities.clone(),points:vec![ObservationSupport{
        path:path.into(),prefill:ObservationSupportStatus::Supported,decode:ObservationSupportStatus::Supported,floating_to_f32:true}]};
    let catalog=ObservationCatalog{schema_version:1,points:vec![point],completeness:DescriptionCompleteness::Complete};
    let selections=if let Some(transform)=transform{vec![CaptureSelection{id:"selected".into(),path:path.into(),schedule:CaptureSchedule::default(),
        slices:vec![],transform}]}else{vec![
        CaptureSelection{id:"full".into(),path:path.into(),schedule:CaptureSchedule::default(),slices:vec![],transform:CaptureTransform::FullTensor},
        CaptureSelection{id:"slice".into(),path:path.into(),schedule:CaptureSchedule::default(),slices:vec![CaptureSlice{
            axis:"vocabulary".into(),start:2,end:6,stride:1}],transform:CaptureTransform::Slice},
        CaptureSelection{id:"preview".into(),path:path.into(),schedule:CaptureSchedule::default(),
            slices:vec![],transform:CaptureTransform::Preview{max_elements:3}}]};
    let usage=CaptureUsage{captures:16,retained_bytes:1<<20,host_bytes:1<<20,encoded_bytes:1<<20};
    let cumulative=usage.checked_mul(attempts).unwrap();
    CapturePlan{schema_version:1,selections,limits:CaptureLimits{per_step:usage,cumulative,
        physical_native_bytes:None,on_limit:CaptureLimitPolicy::Fail}}.admit(&catalog,&support,&capabilities,
            CaptureRequestShape{batch:1,prompt_tokens:1,max_predictions:4}).unwrap()
}
fn values(record:&CaptureRecord)->&[f32]{
    let observation=match record.payload.as_ref().unwrap(){CapturePayload::Tensor(value)=>value,
        CapturePayload::SharedTensor(value)=>value.as_observation(),_=>panic!("raw capture changed payload")};
    let TensorObservationData::F32(values)=observation.data() else{panic!("raw capture changed dtype")};values
}
#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_raw_capture_preserves_full_slice_mask_and_escaped_sources(){
    let artifact=tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();
    let(target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let logits=[-2.5f32,3.75,3.25,-0.125,1.5,50.0,4.0,-4.0];
    let (escaped,evidence)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let(roots,mechanisms)=sources.numerical_prerequisites();
        let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView{shape:&[1,8],values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let input=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&input,&pool,|original|IsolatedArrayCopy::new(original.array())
            .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let input=registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap();
        let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(None,8)).unwrap()).unwrap();
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
        let filter=eredu_core::TokenFilter::Allowed(vec![true,false,true,true,false,true]);
        let mask=eredu_runtime::generation::TokenMaskPlan::new(&filter,input.value().array.shape(),None).unwrap();
        let mut expanded=Vec::with_capacity(mask.elements());mask.fill(&mut expanded).unwrap();
        let ordinary=crate::backend::runtime::generation::apply_token_mask(&MlxTensor::from_array(input.value().array.clone()),
            &expanded,environment.stream()).unwrap();
        let expected=ordinary.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap();drop(ordinary);
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap();
        let mut failed=None;
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],Some(mask),source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("capture lost logits")};
        assert_eq!(value.value().array.evaluated().unwrap().as_slice::<f32>(),expected);
        assert_eq!(evidence.records().len(),3);
        assert_eq!(values(&evidence.records()[0]),logits);
        assert_eq!(values(&evidence.records()[1]),&logits[2..6]);
        assert_eq!(values(&evidence.records()[2]),&logits[..3]);
        assert_eq!(evidence.records()[1].selected_shape.as_deref(),Some([1,1,4].as_slice()));
        // Even a qualified candidate worker must match its original geometry.
        // Refusal leaves the original input and already escaped records intact.
        let unsupported=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(Some(CaptureTransform::TopCandidates{count:2}),9)).unwrap()).unwrap();
        let unsupported=super::super::capture::CaptureSource::original(&unsupported,sources,&pool).unwrap();
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,unsupported,&mut failed);
        assert!(rejected.is_err());drop(rejected);drop(failed);
        assert_eq!(input.value().array.evaluated().unwrap().as_slice::<f32>(),logits);
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        (value,evidence)
    };
    assert!(pool.used_bytes().unwrap()>loaded);
    drop((target,draft));
    assert_eq!(values(&evidence.records()[0]),logits);
    assert_eq!(values(&evidence.records()[1]),&logits[2..6]);
    assert_eq!(values(&evidence.records()[2]),&logits[..3]);
    assert_eq!(escaped.value().array.evaluated().unwrap().as_slice::<f32>(),
        &[-2.5,f32::NEG_INFINITY,3.25,-0.125,f32::NEG_INFINITY,50.0,f32::NEG_INFINITY,f32::NEG_INFINITY]);
    drop((escaped,evidence));drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}

#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_summary_uses_shared_chunk_program_and_retires_failed_and_escaped_sources(){
    use crate::backend::array_copy::{CaptureCompletion,SummaryProgram};
    let artifact=tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();
    let(target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let pattern=[-3.0f32,f32::NAN,f32::INFINITY,f32::NEG_INFINITY,0.0,2.0,4.0,-0.5];
    let mut logits:Vec<f32>=(0..1025).map(|i|pattern[i%pattern.len()]).collect();
    // The last chunk has one zero: count and extrema squeeze aliases execute,
    // while the real zero-scale branch omits its unused moment suffix.
    logits[1024]=0.0;
    let (escaped,evidence,expected)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let(roots,mechanisms)=sources.numerical_prerequisites();
        let shape=[1,logits.len()];
        let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView{shape:&shape,values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let prepared=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&prepared,&pool,|original|IsolatedArrayCopy::new(original.array())
            .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let input=registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap();
        let flat=input.value().array.reshape(&[1025],environment.stream()).unwrap();
        let expected=SummaryProgram::new(1025).unwrap().execute(&flat,environment.stream(),CaptureCompletion::Ordinary,
            &mut |_|Ok(()),&mut |_|{}).unwrap();
        drop(flat);
        let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(
            &declaration(Some(CaptureTransform::Summary),1025)).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap();
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
        let mut failed=None;
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("summary lost logits")};
        assert_eq!(evidence.records().len(),1);
        let CapturePayload::Summary(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("summary changed payload")};
        assert_eq!(actual,&expected);
        assert_eq!((actual.elements,actual.finite,actual.nan,actual.positive_infinity,actual.negative_infinity),(1025,641,128,128,128));
        assert_eq!((actual.min,actual.max),(Some(-3.0),Some(4.0)));
        let finite:Vec<f64>=logits.iter().copied().filter(|v|v.is_finite()).map(f64::from).collect();
        let mean=finite.iter().sum::<f64>()/finite.len() as f64;
        let rms=(finite.iter().map(|v|v*v).sum::<f64>()/finite.len() as f64).sqrt();
        assert!((actual.mean.unwrap()-mean).abs()<1e-6);
        assert!((actual.rms.unwrap()-rms).abs()<1e-6);
        // Exact source geometry remains mandatory even with a complete worker.
        let wrong=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(
            &declaration(Some(CaptureTransform::Summary),1026)).unwrap()).unwrap();
        let wrong=super::super::capture::CaptureSource::original(&wrong,sources,&pool).unwrap();
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,wrong,&mut failed);
        assert!(rejected.is_err());drop(rejected);drop(failed);
        let actual=value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        assert!(actual.iter().zip(&logits).all(|(a,b)|a.to_bits()==b.to_bits()));
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        (value,evidence,expected)
    };
    assert!(pool.used_bytes().unwrap()>loaded);
    drop((target,draft));
    let CapturePayload::Summary(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("escaped summary lost payload")};
    assert_eq!(actual,&expected);
    assert_eq!(escaped.value().array.shape(),[1,1025]);
    drop((escaped,evidence));drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}

#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_histogram_preserves_fixed_edges_counts_and_escaped_sources(){
    use crate::backend::array_copy::{CaptureCompletion,HistogramProgram};
    let artifact=tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();
    let(target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let pattern=[-3.0f32,-2.0,-1.0,-0.0,0.0,1.0,2.0,3.0,4.0,5.0,f32::NAN,f32::INFINITY,f32::NEG_INFINITY,0.5,-1.5,4.0];
    let edges=[-2.0f32,0.0,2.0,4.0];
    let mut logits:Vec<f32>=(0..1025).map(|i|pattern[i%pattern.len()]).collect();
    // Last chunk includes the final upper endpoint and only singleton reductions.
    logits[1024]=4.0;
    let (escaped,evidence)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let(roots,mechanisms)=sources.numerical_prerequisites();
        let shape=[1,logits.len()];
        let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView{shape:&shape,values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let prepared=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&prepared,&pool,|original|IsolatedArrayCopy::new(original.array())
            .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let input=registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap();
        let flat=input.value().array.reshape(&[1025],environment.stream()).unwrap();
        let mut expected_counts=vec![0u64;edges.len()-1];
        let expected=HistogramProgram::new(1025,&edges).unwrap().execute(&flat,environment.stream(),CaptureCompletion::Ordinary,
            &mut |_|Ok(()),&mut |_|{},&mut |index,count|{expected_counts[index]+=count;Ok(())}).unwrap();
        drop(flat);
        let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(
            &declaration(Some(CaptureTransform::Histogram{edges:edges.to_vec()}),1025)).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap();
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
        let mut failed=None;
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("histogram lost logits")};
        assert_eq!(evidence.records().len(),1);
        let CapturePayload::Histogram(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("histogram changed payload")};
        assert_eq!(actual.edges,edges);
        assert_eq!(actual.counts,expected_counts);
        assert_eq!((actual.below,actual.above,actual.non_finite),(expected.below,expected.above,expected.non_finite));
        assert_eq!(actual.counts,[192,256,257]);
        assert_eq!((actual.below,actual.above,actual.non_finite),(64,64,192));
        assert_eq!(actual.counts.iter().sum::<u64>()+actual.below+actual.above+actual.non_finite,1025);
        // Exact source geometry remains mandatory even with a complete worker.
        let wrong=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(
            &declaration(Some(CaptureTransform::Histogram{edges:edges.to_vec()}),1026)).unwrap()).unwrap();
        let wrong=super::super::capture::CaptureSource::original(&wrong,sources,&pool).unwrap();
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,wrong,&mut failed);
        assert!(rejected.is_err());drop(rejected);drop(failed);
        let actual=value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        assert!(actual.iter().zip(&logits).all(|(a,b)|a.to_bits()==b.to_bits()));
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        (value,evidence)
    };
    assert!(pool.used_bytes().unwrap()>loaded);
    drop((target,draft));
    let CapturePayload::Histogram(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("escaped histogram lost payload")};
    assert_eq!(actual.edges,edges);
    assert_eq!(actual.counts,[192,256,257]);
    assert_eq!(escaped.value().array.shape(),[1,1025]);
    drop((escaped,evidence));drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}


#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_token_scores_preserve_distribution_ties_domain_and_escaped_sources(){
    use crate::backend::array_copy::{CaptureCompletion,TokenScoreProgram};
    let artifact=tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();
    let(target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let mut logits:Vec<f32>=(0..1025).map(|i|(i%17) as f32*0.25-2.0).collect();
    logits[1]=7.0;logits[2]=7.0;logits[1024]=-2.0;
    let ids=[1024u32,1,2];
    let (escaped,evidence,expected,partition)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let(roots,mechanisms)=sources.numerical_prerequisites();
        let shape=[1,logits.len()];
        let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView{shape:&shape,values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let prepared=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&prepared,&pool,|original|IsolatedArrayCopy::new(original.array())
            .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let input=registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap();
        let mut allowed=vec![true;1025];allowed[1]=false;
        let filter=eredu_core::TokenFilter::Allowed(allowed);
        let tokenizer=eredu_core::TokenFilter::Allowed(vec![true;1025]);
        let domain=CaptureTokenDomain{filter:(&filter).into(),tokenizer_validity:&tokenizer};
        let mut expected=Vec::new();
        let partition=TokenScoreProgram::new(1025,&ids).unwrap().execute(&input.value().array,environment.stream(),
            CaptureCompletion::Ordinary,&mut |_|Ok(()),&mut |_|{},|id|filter.allows(id),|score|{
                expected.push(score);Ok(())}).unwrap();
        let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(
            Some(CaptureTransform::TokenScores{token_ids:ids.to_vec()}),1025)).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap().with_domain(domain);
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
        let mask=eredu_runtime::generation::TokenMaskPlan::new(&filter,input.value().array.shape(),None).unwrap();
        let mut failed=None;
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],Some(mask),source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("token scoring lost logits")};
        assert_eq!(evidence.records().len(),1);
        let CapturePayload::TokenScores(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("token scoring changed payload")};
        assert_eq!(actual.scores,expected);assert_eq!(actual.log_partition,partition);
        assert_eq!(actual.vocabulary,1025);assert_eq!(actual.domain,Some(domain.summary(1025)));
        assert_eq!(actual.scores.iter().map(|s|s.target.token_id).collect::<Vec<_>>(),ids);
        let independent=logits.iter().map(|&x|(f64::from(x)-7.0).exp()).sum::<f64>().ln()+7.0;
        assert!((partition-independent).abs()<1e-6);
        for score in &actual.scores {
            let id=score.target.token_id as usize;
            assert_eq!(score.target.score,logits[id]);assert_eq!(score.target.allowed,id!=1);
            assert_eq!(score.rank,1+logits.iter().filter(|&&x|x>logits[id]).count() as u64);
            assert!((score.log_probability-(f64::from(logits[id])-independent)).abs()<1e-6);
            let alternative=score.strongest_alternative.as_ref().unwrap();
            assert_eq!(alternative.token_id,if id==1{2}else{1});
            assert_eq!(alternative.score,7.0);assert_eq!(alternative.allowed,id==1);
        }
        let wrong=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(
            Some(CaptureTransform::TokenScores{token_ids:ids.to_vec()}),1026)).unwrap()).unwrap();
        let wrong=super::super::capture::CaptureSource::original(&wrong,sources,&pool).unwrap();
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,wrong,&mut failed);
        assert!(rejected.is_err());drop(rejected);drop(failed);
        assert_eq!(input.value().array.evaluated().unwrap().as_slice::<f32>(),logits);
        let output=value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        for (id,&score) in output.iter().enumerate(){assert_eq!(score,if id==1{f32::NEG_INFINITY}else{logits[id]});}
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        (value,evidence,expected,partition)
    };
    assert!(pool.used_bytes().unwrap()>loaded);drop((target,draft));
    let CapturePayload::TokenScores(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("escaped scores lost payload")};
    assert_eq!(actual.scores,expected);assert_eq!(actual.log_partition,partition);
    assert_eq!(escaped.value().array.shape(),[1,1025]);
    drop((escaped,evidence));drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}


#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_candidates_preserve_full_sort_ties_domains_failure_and_escaped_sources(){
    use crate::backend::array_copy::{CaptureCompletion,CandidateExtraction};
    let artifact=tempfile::tempdir().unwrap();crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();let(target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let mut logits:Vec<f32>=(0..4097).map(|i|(i%17) as f32*0.25-2.0).collect();
    logits[1]=7.0;logits[2]=7.0;logits[4096]=7.0;
    let escaped={
        let mut escaped=Vec::new();
        // A request binds its cumulative capture ledger to one immutable source.
        // Different candidate counts are distinct admissions, each with a real
        // request; escaped outputs retain their original request custody.
        for count in [3u64,4097] {
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();let sources=pair.numerical_sources();
        let(roots,mechanisms)=sources.numerical_prerequisites();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let make_input=|data:&[f32]| {
            let shape=[1,data.len()];
            let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
                payload:HostTensorView{shape:&shape,values:HostTensorValues::F32(data)},metadata:&[],extents:&[]}];
            let prepared=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
            let copy=materializer.with_test_original_copy_source(&prepared,&pool,|original|IsolatedArrayCopy::new(original.array())
                .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
            registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap()
        };
        let input=make_input(&logits);
        let mut allowed=vec![true;4097];allowed[2]=false;
        let filter=eredu_core::TokenFilter::Allowed(allowed);
        let tokenizer=eredu_core::TokenFilter::Allowed(vec![true;4097]);
        let domain=CaptureTokenDomain{filter:(&filter).into(),tokenizer_validity:&tokenizer};
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
            let row=input.value().array.reshape(&[1,1,4097],environment.stream()).unwrap();
            let program=CandidateExtraction::borrowed(row.shape(),count).unwrap();
            let(ids,scores)=program.execute_with_completion(&row,environment.stream(),CaptureCompletion::Ordinary,&mut |_|Ok(())).unwrap();
            let mut expected=Vec::new();program.read_with_completion(&ids,&scores,environment.stream(),CaptureCompletion::Ordinary,
                |id,score|{expected.push(CaptureCandidate{token_id:id,score,allowed:filter.allows(id)});Ok(())}).unwrap();
            drop((row,ids,scores));
            // This immutable source declares the successful capture and the
            // later nonfinite attempt. Both consume the same cumulative ledger.
            let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration_with_attempts(
                Some(CaptureTransform::TopCandidates{count}),4097,2)).unwrap()).unwrap();
            let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap().with_domain(domain);
            let mask=eredu_runtime::generation::TokenMaskPlan::new(&filter,input.value().array.shape(),None).unwrap();
            let mut failed=None;
            let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],Some(mask),source,&mut failed).unwrap();
            assert!(failed.is_none());
            let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("candidate capture lost logits")};
            let CapturePayload::Candidates(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("candidates changed payload")};
            assert_eq!(actual.candidates,expected);assert_eq!(actual.domain,Some(domain.summary(4097)));
            assert_eq!(actual.candidates.len(),count as usize);
            assert_eq!(actual.candidates[..3].iter().map(|c|c.token_id).collect::<Vec<_>>(),[4096,2,1]);
            let mut independent:Vec<usize>=(0..logits.len()).collect();
            independent.sort_by(|&a,&b|logits[b].partial_cmp(&logits[a]).unwrap().then(b.cmp(&a)));
            for (candidate,&id) in actual.candidates.iter().zip(&independent) {
                assert_eq!(candidate.token_id,id as u32);assert_eq!(candidate.score,logits[id]);assert_eq!(candidate.allowed,id!=2);
            }
            let output=value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap();
            for(id,&score)in output.iter().enumerate(){assert_eq!(score,if id==2{f32::NEG_INFINITY}else{logits[id]});}
            escaped.push((value,evidence,expected));
        // A real nonfinite row refuses after the finite-count read, before the
        // sort. Retaining the original failed input must not alter prior records.
        let mut nonfinite=logits.clone();nonfinite[0]=f32::NAN;let bad=make_input(&nonfinite);
        // Reuse the same source so rejection reaches the finite-score worker.
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap();let mut failed=None;
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&bad,&[],None,source,&mut failed);
        let rejected=rejected.expect_err("nonfinite candidate row must refuse");
        let mut cause:Option<&(dyn std::error::Error+'static)>=Some(&rejected);
        let mut finite_refusal=false;
        while let Some(error)=cause {
            finite_refusal|=error.to_string()=="candidate capture requires finite raw logits";
            cause=error.source();
        }
        assert!(finite_refusal,"wrong candidate failure: {rejected:?}");
        drop(rejected);drop(failed);
        assert!(bad.value().array.evaluated().unwrap().as_slice::<f32>()[0].is_nan());
        assert_eq!(input.value().array.evaluated().unwrap().as_slice::<f32>(),logits);
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        }
        escaped
    };
    assert!(pool.used_bytes().unwrap()>loaded);drop((target,draft));
    for(value,evidence,expected)in &escaped {
        let CapturePayload::Candidates(actual)=evidence.records()[0].payload.as_ref().unwrap() else{panic!("escaped candidates lost payload")};
        assert_eq!(actual.candidates,*expected);assert_eq!(value.value().array.shape(),[1,4097]);
    }
    drop(escaped);drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}

#[test]
#[ignore="requires managed Metal allocator and CPU execution"]
fn original_cpu_scale_interventions_preserve_order_evidence_partial_regions_and_escaped_sources() {
    use eredu_core::intervention::*;
    use crate::composition::mlx::session::bounded_capture::NativeCapture;
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let cpu = MlxBackend::for_prepared_execution_plan(PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial = pool.used_bytes().unwrap();
    let (target_config,draft_config,selected) = source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config); let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig{max_tokens:4,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let logits:Vec<f32>=(0..19).map(|i|(i as f32-7.0)*0.25).collect();
    let intervention_plan=|partial:bool| {
        let path=eredu_core::MODEL_LOGITS_OBSERVATION_PATH;
        let declaration=InterventionDiscovery{schema_version:INTERVENTION_SCHEMA_VERSION,artifact_identity:"model".into(),
            session_identity:Some("session".into()),points:vec![InterventionPoint{path:path.into(),node_id:"actual.cpu.logits".into(),
                stage:InterventionStage::LogitsBeforeSampling,
                axes:[("batch",SymbolicDimension::Batch),("sequence",SymbolicDimension::Sequence),
                    ("vocabulary",SymbolicDimension::Known(19))].into_iter()
                    .map(|(name,dimension)|TensorAxis{name:name.into(),dimension}).collect(),
                dtypes:vec![InterventionDtype::Float32],operations:vec![InterventionKind::Scale],score_stages:vec![],
                prefill:ObservationSupportStatus::Supported,decode:ObservationSupportStatus::Supported,
                conditions:vec![],routing:None,routed_units:None}]};
        InterventionPlan{schema_version:INTERVENTION_SCHEMA_VERSION,operations:[(0.5,InterventionEvidence::Preview{max_elements:3}),
            (-2.0,InterventionEvidence::Summary)].into_iter().enumerate().map(|(i,(factor,evidence))|InterventionOperation{
                id:i.to_string(),target:path.into(),schedule:Default::default(),
                slices:if partial {vec![CaptureSlice{axis:"vocabulary".into(),start:1,end:3,stride:1}]}else{vec![]},
                action:InterventionAction::Scale{dtype:InterventionDtype::Float32,factor},evidence,
            }).collect()}.admit(&declaration,CaptureRequestShape{batch:1,prompt_tokens:1,max_predictions:4},"session").unwrap()
    };
    let (escaped,evidence,partial_escaped,partial_evidence,expected,partial_expected) = {
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),draft.original_model_source().unwrap(),
            &schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources(); let (roots,mechanisms)=sources.numerical_prerequisites();
        let shape=[1,logits.len()];
        let parts=[HostInputPart{modality:eredu_core::InputModality::Text,kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView{shape:&shape,values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let prepared=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&prepared,&pool,|original|IsolatedArrayCopy::new(original.array())
            .copy_prepared(&original,&environment,roots,mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let input=registered::value_input(copy,Meaning::Logits,sources,&environment).unwrap();
        let plan=intervention_plan(false);
        // Ordinary and admitted execution use the same neutral action sequence
        // and the same static native operations, including the full update alias.
        let original=input.value().array.reshape(&[1,1,19],environment.stream()).unwrap();
        let mut ordinary=MlxTensor::from_array(original);
        let mut worker=NativeCapture{stream:environment.stream(),domain:None,partition:None};
        let slice=ResolvedCaptureSlice{starts:vec![0;3],ends:vec![1,1,19],strides:vec![1;3],shape:vec![1,1,19]};
        for operation in &plan.plan().operations {
            ordinary=eredu_runtime::intervention::apply_activation(&mut worker,&ordinary,&operation.action,&slice).unwrap();
        }
        let expected=ordinary.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap(); drop(ordinary);
        assert_eq!(expected,logits.iter().map(|v|-v).collect::<Vec<_>>());
        let capture=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(Some(CaptureTransform::FullTensor),19)).unwrap()).unwrap();
        let interventions=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&plan).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap().with_interventions(&interventions);
        let policy=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&DefaultSampler)
            .unwrap().bind(0.0,0).unwrap();
        let mut failed=None;
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value,capture:evidence}=result else{panic!("intervention lost captured logits")};
        assert_eq!(value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
        assert_eq!(values(&evidence.records()[0]),logits);
        assert_eq!(evidence.interventions().len(),2);
        for record in evidence.interventions() {
            assert_eq!(record.outcome,InterventionOutcome::Applied);
            assert_eq!(record.evidence.len(),2);
            assert!(record.charged.retained_bytes>0);
        }
        let first=&evidence.interventions()[0];
        assert_eq!(values(&first.evidence[0]),&logits[..3]);
        assert_eq!(values(&first.evidence[1]),&logits[..3].iter().map(|v|v*0.5).collect::<Vec<_>>());
        let second=&evidence.interventions()[1];
        for (record,factor) in second.evidence.iter().zip([0.5,-1.0]) {
            let CapturePayload::Summary(summary)=record.payload.as_ref().unwrap() else{panic!("summary evidence changed kind")};
            assert_eq!(summary.finite,19);
            let expected_mean=logits.iter().map(|v|f64::from(*v)*factor).sum::<f64>()/19.0;
            assert!((summary.mean.unwrap()-expected_mean).abs()<1e-6);
        }
        // The same ordered actions now replace only the selected interval.
        // Compute the ordinary result through the same neutral/static worker,
        // then compare with an independent coordinate-based expected result.
        let partial_plan=intervention_plan(true);
        let original=input.value().array.reshape(&[1,1,19],environment.stream()).unwrap();
        let mut ordinary=MlxTensor::from_array(original);
        let slice=ResolvedCaptureSlice{starts:vec![0,0,1],ends:vec![1,1,3],strides:vec![1;3],shape:vec![1,1,2]};
        for operation in &partial_plan.plan().operations {
            ordinary=eredu_runtime::intervention::apply_activation(&mut worker,&ordinary,&operation.action,&slice).unwrap();
        }
        let partial_expected=ordinary.as_array().evaluated().unwrap().try_to_vec::<f32>().unwrap();drop(ordinary);
        let mut coordinates=logits.clone();for value in &mut coordinates[1..3]{*value = -*value;}
        assert_eq!(partial_expected,coordinates);
        let partial=pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&partial_plan).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&capture,sources,&pool).unwrap().with_interventions(&partial);
        let result=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,source,&mut failed).unwrap();
        assert!(failed.is_none());
        let NumericalOutput::CapturedLogits{value:partial_value,capture:partial_evidence}=result else{panic!("partial intervention lost captured logits")};
        assert_eq!(partial_value.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),partial_expected);
        assert_eq!(values(&partial_evidence.records()[0]),logits);
        assert_eq!(partial_evidence.interventions().len(),2);
        for record in partial_evidence.interventions(){
            assert_eq!(record.outcome,InterventionOutcome::Applied);assert_eq!(record.evidence.len(),2);
        }
        let first=&partial_evidence.interventions()[0];
        assert_eq!(values(&first.evidence[0]),&logits[1..3]);
        assert_eq!(values(&first.evidence[1]),&logits[1..3].iter().map(|v|v*0.5).collect::<Vec<_>>());
        for (record,factor) in partial_evidence.interventions()[1].evidence.iter().zip([0.5,-1.0]){
            let CapturePayload::Summary(summary)=record.payload.as_ref().unwrap() else{panic!("partial summary changed kind")};
            assert_eq!(summary.finite,2);
            let mean=logits[1..3].iter().map(|v|f64::from(*v)*factor).sum::<f64>()/2.0;
            assert!((summary.mean.unwrap()-mean).abs()<1e-6);
        }
        // A declaration for another source width remains a real source refusal;
        // successful full/partial outputs and their evidence retain ownership.
        let wrong=pool.compile_capture_source(PreparedCapturePlanCopy::inspect(&declaration(Some(CaptureTransform::FullTensor),20)).unwrap()).unwrap();
        let source=super::super::capture::CaptureSource::original(&wrong,sources,&pool).unwrap().with_interventions(&partial);
        let rejected=NumericalProducer::execute_captured_policy(sources,&environment,roots,mechanisms,policy,&input,&[],None,source,&mut failed);
        assert!(rejected.is_err());drop(rejected);drop(failed);
        assert_eq!(input.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),logits);
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        (value,evidence,partial_value,partial_evidence,expected,partial_expected)
    };
    assert!(pool.used_bytes().unwrap()>loaded);
    drop((target,draft));
    assert_eq!(escaped.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
    assert_eq!(values(&evidence.interventions()[0].evidence[0]),&logits[..3]);
    assert_eq!(partial_escaped.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),partial_expected);
    assert_eq!(values(&partial_evidence.interventions()[0].evidence[0]),&logits[1..3]);
    drop((escaped,evidence,partial_escaped,partial_evidence)); drop(schedule); drop((target_config,draft_config,selected)); settle(&pool,initial);
}
