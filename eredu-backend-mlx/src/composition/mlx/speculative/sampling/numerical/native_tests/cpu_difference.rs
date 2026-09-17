//! Actual two-cut CPU correction from completed registered F32 inputs.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::input::host::{HostInputPart,HostTensorValues,HostTensorView,PreparedHostInputPlan};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_correction_matches_mass_branches_and_retires_escaped_output() {
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
    let other=[1.5f32,0.1,-0.5,3.0,-2.0];
    let (escaped,expected)={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),&schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let (roots,mechanisms)=sources.numerical_prerequisites();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let source=|values:&[f32]|{
            let parts=[HostInputPart {modality:eredu_core::InputModality::Text,
                kind:eredu_core::InputPayloadKind::Embeddings,
                payload:HostTensorView {shape:&[1,1,5],values:HostTensorValues::F32(values)},metadata:&[],extents:&[]}];
            let input=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
            let copy=materializer.with_test_original_copy_source(&input,&pool,|original|
                IsolatedArrayCopy::new(original.array()).copy_prepared(&original,&environment,roots,
                    mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
            crate::composition::mlx::speculative::sampling::numerical::registered::value_input(
                copy,Meaning::Logits,sources,&environment).unwrap()
        };
        let left=source(&logits);let right=source(&other);
        let (ordinary,mass)=program::correction::<operators::Native>(&left.value().array,&right.value().array,environment.stream()).unwrap();
        let mass=mass.evaluated().unwrap().try_item::<f32>().unwrap();assert!(mass>f32::EPSILON);
        let ordinary=program::correction_logits::<operators::Native>(&ordinary,environment.stream()).unwrap();
        let expected=ordinary.evaluated().unwrap().try_to_vec::<f32>().unwrap();drop(ordinary);
        assert!(expected.iter().any(|v|v.is_finite()));assert!(expected.contains(&f32::NEG_INFINITY));
        let output=NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::Correction,&left,Some(&right)).unwrap();
        let NumericalOutput::Correction(Some(output))=output else{panic!("positive mass lost corrected logits")};
        assert_eq!(output.value().meaning,Meaning::Logits);
        assert_eq!(output.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
        assert!(output.value().provenance.source().belongs_to_request(pair.request()));
        // Identical inputs take the same completed zero-mass branch. They do
        // not construct or evaluate logarithmic negative-infinity output.
        let zero=NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::Correction,&left,Some(&left)).unwrap();
        assert!(matches!(zero,NumericalOutput::Correction(None)));
        pair.request().close().unwrap();
        assert!(NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::Correction,&left,Some(&right)).is_err());
        (output,expected)
    };
    assert!(pool.used_bytes().unwrap()>loaded);assert_eq!(pool.unquoted_owner_count().unwrap(),0);
    drop((target,draft));
    assert_eq!(escaped.value().array.evaluated().unwrap().try_to_vec::<f32>().unwrap(),expected);
    drop(escaped);drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}
