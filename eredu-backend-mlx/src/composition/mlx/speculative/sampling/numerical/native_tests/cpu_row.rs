//! Actual CPU row selection retains its full registered source through sampling.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::{DefaultSampler,SpeculativeSampler};
use eredu_runtime::input::host::{HostInputPart,HostTensorValues,HostTensorView,PreparedHostInputPlan};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_row_readout_preserves_selected_values_and_escaped_parent() {
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
    let config=SpeculativeConfig {max_tokens:2,max_draft_tokens:1,temperature:0.0,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let logits=[50.0f32,2.0,1.0,-1.0,0.0,-4.0,0.5,3.0,8.0,8.0];
    let escaped={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),&schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let sources=pair.numerical_sources();let (roots,mechanisms)=sources.numerical_prerequisites();
        let parts=[HostInputPart {modality:eredu_core::InputModality::Text,
            kind:eredu_core::InputPayloadKind::Embeddings,
            payload:HostTensorView {shape:&[1,2,5],values:HostTensorValues::F32(&logits)},metadata:&[],extents:&[]}];
        let input=pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let materializer=MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy=materializer.with_test_original_copy_source(&input,&pool,|original|
            IsolatedArrayCopy::new(original.array()).copy_prepared(&original,&environment,roots,
                mechanisms,sources.metadata_funding(),REQUEST_CEILING).unwrap());
        let source=crate::composition::mlx::speculative::sampling::numerical::registered::value_input(
            copy,Meaning::Logits,sources,&environment).unwrap();
        let backing=source.value().array.try_allocation_info().unwrap().unwrap().identity();
        let ordinary=crate::composition::mlx::prepared_speculative::embedded_logits::row(
            &source.value().array,1,environment.stream()).unwrap();
        let expected=ordinary.evaluated().unwrap().as_slice::<f32>().to_vec();drop(ordinary);
        assert_eq!(expected,&logits[5..]);
        let output=NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::LogitsRow {row:1},&source,None).unwrap();
        let NumericalOutput::Logits(row)=output else {panic!("row lost logit meaning")};
        assert_eq!(row.value().array.shape(),&[1,5]);
        assert_eq!(row.value().array.evaluated().unwrap().as_slice::<f32>(),expected);
        assert_eq!(row.value().array.try_allocation_info().unwrap().unwrap().identity(),backing);
        let choice=<DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>
            ::prepared_greedy_policy(&DefaultSampler).unwrap().greedy(0.0).unwrap();
        let selected=NumericalProducer::execute(sources,&environment,roots,mechanisms,
            program::SpeculativeNumericalKind::Greedy(choice),&row,None).unwrap();
        let NumericalOutput::Token(token)=selected else {panic!("row greedy lost token meaning")};
        assert_eq!(token,3);
        assert!(row.value().provenance.source().belongs_to_request(pair.request()));
        drop(source);drop(input);
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);
        row
    };
    assert!(pool.used_bytes().unwrap()>loaded);
    drop((target,draft));
    assert_eq!(escaped.value().array.evaluated().unwrap().as_slice::<f32>(),&logits[5..]);
    drop(escaped);drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}
