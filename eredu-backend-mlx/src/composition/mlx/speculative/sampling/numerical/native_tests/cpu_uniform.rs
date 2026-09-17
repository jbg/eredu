//! CPU acceptance draws use the same explicit-key native uniform equation.
use super::*;

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_uniform_matches_ordinary_and_preserves_failed_key_and_custody() {
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
    let config=SpeculativeConfig {max_tokens:4,max_draft_tokens:1,temperature:0.7,eos_token_ids:Vec::new()};
    let schedule=AutoregressiveSchedulePlan::new(&selected,NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),NonZeroU64::new(32).unwrap(),&config,SpeculativeSchedulerOptions::default()).unwrap();
    let mut ordinary=RandomState::with_seed(SEED).unwrap();
    let expected=std::array::from_fn::<_,3,_>(|_| {
        let draw=ordinary.uniform_unit_interval(cpu.stream()).unwrap();
        let value=draw.evaluated().unwrap().as_slice::<f32>()[0];
        assert!(value>0.0&&value<1.0);
        (value,ordinary_words(ordinary.as_array()))
    });
    assert_ne!(expected[0].0,expected[1].0);drop(ordinary);
    let escaped={
        let pair=AutoregressiveSourcePair::prepare(target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),&schedule,&pool,REQUEST_CEILING).unwrap();
        let environment=cpu.original_copy_environment().unwrap();
        let context=SpeculativeExecutionStreams::single(cpu.stream())
            .with_original_numerical_sources(pair.numerical_sources(),&environment).unwrap();
        let mut state=create_key(SEED,context).unwrap();
        for (value,words) in expected {
            assert_eq!(sample_unit_interval(&mut state,context).unwrap(),value);
            assert_eq!(key_words(&state),words);
        }
        pair.request().close().unwrap();let saved=key_words(&state);
        let error=sample_unit_interval(&mut state,context).unwrap_err();
        assert_eq!(key_words(&state),saved);drop(error);
        assert_eq!(pool.unquoted_owner_count().unwrap(),0);state
    };
    assert!(pool.used_bytes().unwrap()>loaded);drop((target,draft));
    assert_eq!(key_words(&escaped),expected[2].1);
    drop(escaped);drop(schedule);drop((target_config,draft_config,selected));settle(&pool,initial);
}
