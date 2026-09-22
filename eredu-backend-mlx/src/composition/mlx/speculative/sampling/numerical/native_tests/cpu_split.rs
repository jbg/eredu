//! Shared CPU split/row workers preserve PRNG state and completed source custody.
use super::*;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_split_and_position_match_ordinary_and_preserve_failed_state() {
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
    let mut ordinary = RandomState::with_seed(SEED).unwrap();
    let ordinary_next = ordinary.next_key(cpu.stream()).unwrap();
    let expected = [
        ordinary_words(ordinary.as_array()),
        ordinary_words(&ordinary_next),
        ordinary_words(
            &split_key_at(&safemlx::random::key(SEED).unwrap(), 3, cpu.stream()).unwrap(),
        ),
    ];
    drop((ordinary, ordinary_next));
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
        let context = SpeculativeExecutionStreams::single(cpu.stream())
            .with_original_numerical_sources(pair.numerical_sources(), &environment)
            .unwrap();
        let root = create_key(SEED, context).unwrap();
        let mut state = root.clone();
        let next = next_key(&mut state, context).unwrap();
        let at = key_at(&root, SpeculativeDraftRandomPosition::new(3), context).unwrap();
        assert_eq!(
            [key_words(&state), key_words(&next), key_words(&at)],
            expected
        );
        assert_ne!(expected[0], expected[1]);
        assert_eq!(key_words(&root), [(SEED >> 32) as u32, SEED as u32]);
        let current = state
            .value()
            .value()
            .array
            .try_allocation_info()
            .unwrap()
            .unwrap()
            .identity();
        assert_eq!(
            current,
            next.value()
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        );
        assert_ne!(
            current,
            root.value()
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        );
        pair.request().close().unwrap();
        let saved = key_words(&state);
        let error = next_key(&mut state, context).unwrap_err();
        assert_eq!(key_words(&state), saved);
        drop(error);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        [state, next, at]
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    drop((target, draft));
    assert_eq!(escaped.each_ref().map(key_words), expected);
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
