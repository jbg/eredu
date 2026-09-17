//! Actual prepared source, CPU copy/complete and existing registered receiver.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan};

#[test]
#[ignore = "requires managed native Metal allocator and CPU execution"]
fn original_cpu_copy_preserves_prepared_source_and_escaped_custody() {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let cpu_streams = PreparedExecutionStreams::for_cpu_factory(&pool).unwrap().unwrap();
    let identity = MlxDeviceIdentity::from_realized_device(
        &safemlx::Device::new(safemlx::DeviceType::Cpu, 0), None,
    ).unwrap();
    let cpu = MlxBackend::for_prepared_execution_plan(cpu_streams, identity);
    let initial = pool.used_bytes().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.used_bytes().unwrap();
    let config = SpeculativeConfig { max_tokens: 2, max_draft_tokens: 1,
        temperature: 0.0, eos_token_ids: Vec::new() };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected, NonZeroUsize::new(1).unwrap(), NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(), &config, SpeculativeSchedulerOptions::default(),
    ).unwrap();
    let values = [2i32, -7, i32::MIN, 19, 5];
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(), draft.original_model_source().unwrap(),
            &schedule, &pool, REQUEST_CEILING,
        ).unwrap();
        let environment = cpu.original_copy_environment().unwrap();
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let parts = [HostInputPart {
            modality: eredu_core::InputModality::Text,
            kind: eredu_core::InputPayloadKind::TokenIds,
            payload: HostTensorView { shape: &[1, 5], values: HostTensorValues::I32(&values) },
            metadata: &[], extents: &[],
        }];
        let input = pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap();
        let copied = materializer.with_test_original_copy_source(&input, &pool, |original| {
            let source = original.array().try_allocation_info().unwrap().unwrap().identity();
            let copy = IsolatedArrayCopy::new(original.array()).copy_prepared(
                &original, &environment, roots, mechanisms, sources.metadata_funding(), REQUEST_CEILING,
            ).unwrap();
            copy.validate_completed_stream(environment.stream(), sources.metadata_funding()).unwrap();
            assert_ne!(copy.array().try_allocation_info().unwrap().unwrap().identity(), source);
            copy
        });
        let output = registered_copy_input(copied, sources, &environment).unwrap();
        let proof = RegisteredTensorSource::from_value(&output, sources, &environment).unwrap();
        proof.validate_array(&output.value().array, sources.metadata_funding()).unwrap();
        drop(input);
        output
    };
    assert!(pool.used_bytes().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    assert_eq!(escaped.value().array.evaluated().unwrap().as_slice::<i32>(), values);
    assert_eq!(escaped.value().array.shape(), &[1, 5]);
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
