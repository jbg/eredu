//! Shared eager seed construction with actual CPU completion and escaped custody.
use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_key_preserves_unsigned_seed_words_and_escaped_custody() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let cpu = MlxBackend::for_prepared_execution_plan(
        PreparedExecutionStreams::for_cpu_factory(&pool)
            .unwrap()
            .unwrap(),
        MlxDeviceIdentity::from_realized_device(
            &safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
            None,
        )
        .unwrap(),
    );
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
    let seeds = [0u64, SEED, u64::MAX];
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            crate::memory_fixture::resolved_limits(REQUEST_CEILING),
        )
        .unwrap();
        let environment = cpu.original_copy_environment().unwrap();
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let keys = seeds.map(|seed| {
            let ordinary = safemlx::random::key(seed).unwrap();
            let expected = ordinary_words(&ordinary);
            drop(ordinary);
            let output = NumericalProducer::execute_inputs(
                sources,
                &environment,
                roots,
                mechanisms,
                program::SpeculativeNumericalKind::CreateKey { seed },
                None,
                None,
                None,
            )
            .unwrap();
            let NumericalOutput::Key(key) = output else {
                panic!("seed constructor lost key meaning")
            };
            assert_eq!(key_words(&key), expected);
            assert_eq!(expected, [(seed >> 32) as u32, seed as u32]);
            assert!(key
                .value()
                .value()
                .provenance
                .source()
                .belongs_to_request(pair.request()));
            assert!(key
                .value()
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .is_some());
            key
        });
        assert_ne!(
            keys[0]
                .value()
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity(),
            keys[1]
                .value()
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        );
        pair.request().close().unwrap();
        let refusal = NumericalProducer::execute_inputs(
            sources,
            &environment,
            roots,
            mechanisms,
            program::SpeculativeNumericalKind::CreateKey { seed: 17 },
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(std::error::Error::source(&refusal).is_some());
        drop(refusal);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        keys
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    drop((target, draft));
    for (key, seed) in escaped.iter().zip(seeds) {
        assert_eq!(key_words(key), [(seed >> 32) as u32, seed as u32]);
    }
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
