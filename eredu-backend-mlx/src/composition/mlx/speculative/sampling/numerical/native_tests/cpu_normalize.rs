//! Actual CPU numerical Normalize from completed registered F32 input.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::input::host::{HostInputPart,HostTensorValues,HostTensorView,PreparedHostInputPlan};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_normalization_matches_ordinary_and_retires_escaped_output() {
    let artifact=tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool=crate::tests::support::test_utils::initialize_original_sources();
    let backend=admitted_backend(&pool);
    let cpu=MlxBackend::for_prepared_execution_plan(
        PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap(),
        MlxDeviceIdentity::from_realized_device(&safemlx::Device::new(safemlx::DeviceType::Cpu,0),None).unwrap());
    let initial=pool.used_bytes().unwrap();
    let (target_config,draft_config,selected)=source_configs(&backend,artifact.path());
    let target=load(&backend,&target_config);let draft=load(&backend,&draft_config);
    let loaded=pool.used_bytes().unwrap();
    let config=SpeculativeConfig {max_tokens:2,max_draft_tokens:1,temperature:0.7,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let logits=[-2.5f32,0.25,3.75,-0.125,1.5];
    let (escaped,expected)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),&schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let (roots,mechanisms)=sources.numerical_prerequisites();
        let parts=[HostInputPart {modality:eredu_core::InputModality::Text,
            kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView {shape:&[1,1,5],values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let input=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&input,&pool,|original|
            IsolatedArrayCopy::new(original.array()).copy_prepared(&original,&environment,roots,
                mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let source=crate::composition::mlx::speculative::sampling::numerical::registered::value_input(
            copy,Meaning::Logits,sources,&environment).unwrap();
        let ordinary=program::normalize::<operators::Native>(&source.value().array,environment.stream()).unwrap();
        let expected=ordinary.evaluated().unwrap().as_slice::<f32>().to_vec();drop(ordinary);
        let output=NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::Normalize,&source,None).unwrap();
        let NumericalOutput::Distribution(output)=output else {panic!("normalization lost probability meaning")};
        assert_eq!(output.value().meaning,Meaning::Probabilities);
        assert_eq!(output.value().array.shape(),&[1,1,5]);
        assert_eq!(output.value().array.evaluated().unwrap().as_slice::<f32>(),expected);
        assert!(expected.iter().all(|p|*p>0.0));assert!((expected.iter().sum::<f32>()-1.0).abs()<1e-6);
        assert!(output.value().provenance.source().belongs_to_request(pair.request()));
        // Distribution custody is carried by its completed native allocation
        // and numerical account; the copy-source convenience field is optional.
        assert!(output.value().array.try_allocation_info().unwrap().is_some());
        drop(source);drop(input);(output,expected)
    };
    assert!(pool.used_bytes().unwrap()>loaded);assert_eq!(pool.unquoted_owner_count().unwrap(),0);
    drop((target,draft));
    assert_eq!(escaped.value().array.evaluated().unwrap().as_slice::<f32>(),expected);
    drop(escaped);drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}
