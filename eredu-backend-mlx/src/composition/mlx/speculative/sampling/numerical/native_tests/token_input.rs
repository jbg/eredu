//! Exact eager U32 upload through the real numerical source/account/completion path.
use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;

#[test]
#[ignore = "requires native Metal execution"]
fn original_numerical_token_input_preserves_unsigned_bits_and_escaped_custody() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    token_input_case(false);
}
#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_numerical_token_input_and_range_preserve_unsigned_bits_and_escaped_custody() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    token_input_case(true);
}
fn token_input_case(use_cpu: bool) {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let cpu = use_cpu.then(|| {
        let streams = PreparedExecutionStreams::for_cpu_factory(&pool)
            .unwrap()
            .unwrap();
        let identity = MlxDeviceIdentity::from_realized_device(
            &safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
            None,
        )
        .unwrap();
        MlxBackend::for_prepared_execution_plan(streams, identity)
    });
    let initial = pool.fixture_host_charge().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.fixture_host_charge().unwrap();
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: 0.7,
        eos_token_ids: Vec::new(),
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let ids = [0, u32::MAX, 17, (i32::MAX as u32) + 1];
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            crate::memory_fixture::resolved_limits(REQUEST_CEILING),
        )
        .unwrap();
        let environment = cpu
            .as_ref()
            .unwrap_or(&backend)
            .original_copy_environment()
            .unwrap();
        assert_eq!(
            environment.stream().device_type().unwrap(),
            if use_cpu {
                safemlx::DeviceType::Cpu
            } else {
                safemlx::DeviceType::Gpu
            }
        );
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let output =
            NumericalProducer::execute_token_ids(sources, &environment, roots, mechanisms, &ids)
                .unwrap();
        let NumericalOutput::Tensor(value) = output else {
            panic!("token input lost tensor meaning");
        };
        let value = {
            if use_cpu {
                let backing = value
                    .value()
                    .array
                    .try_allocation_info()
                    .unwrap()
                    .unwrap()
                    .identity();
                let output = NumericalProducer::execute(
                    sources,
                    &environment,
                    roots,
                    mechanisms,
                    program::SpeculativeNumericalKind::TokenRange { start: 1, end: 3 },
                    &value,
                    None,
                )
                .unwrap();
                let NumericalOutput::Tensor(range) = output else {
                    panic!("CPU range lost tensor meaning");
                };
                assert_eq!(range.value().meaning, Meaning::TokenIds);
                assert_eq!(range.value().array.shape(), &[1, 2]);
                assert_eq!(
                    range.value().array.evaluated().unwrap().as_slice::<u32>(),
                    &ids[1..3]
                );
                assert_eq!(
                    range
                        .value()
                        .array
                        .try_allocation_info()
                        .unwrap()
                        .unwrap()
                        .identity(),
                    backing
                );
                assert!(range
                    .value()
                    .provenance
                    .source()
                    .belongs_to_request(pair.request()));
                // Escaping a view retains both the old backing and current view role.
                drop(value);
                range
            } else {
                value
            }
        };
        assert_eq!(value.value().meaning, Meaning::TokenIds);
        assert_eq!(
            value.value().array.shape(),
            if use_cpu { &[1, 2][..] } else { &[1, 4][..] }
        );
        assert_eq!(value.value().array.dtype(), safemlx::Dtype::Uint32);
        assert!(value.value().original_budget.is_some());
        assert!(value
            .value()
            .provenance
            .source()
            .belongs_to_request(pair.request()));
        assert_eq!(
            value.value().array.evaluated().unwrap().as_slice::<u32>(),
            if use_cpu { &ids[1..3] } else { &ids[..] }
        );
        value
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    // The completed value owns its actual native backing and phase account
    // after the request and both executable instances retire.
    assert_eq!(
        escaped.value().array.evaluated().unwrap().as_slice::<u32>(),
        if use_cpu { &ids[1..3] } else { &ids[..] }
    );
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
