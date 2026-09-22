//! Actual CPU known-policy categorical from completed registered scores.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::MlxPreparedInputMaterializer;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use eredu_runtime::{DefaultSampler, SamplingBackend};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_categorical_matches_ordinary_and_preserves_key_failure_and_custody() {
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
        max_tokens: 4,
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
    let logits = [-2.5f32, 0.25, 3.75, -0.125, 1.5];
    let (escaped, expected) = {
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
        let parts = [HostInputPart {
            modality: eredu_core::InputModality::Text,
            kind: eredu_core::InputPayloadKind::Embeddings,
            payload: HostTensorView {
                shape: &[1, 1, 5],
                values: HostTensorValues::F32(&logits),
            },
            metadata: &[],
            extents: &[],
        }];
        let input = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
            .unwrap();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy = materializer.with_test_original_copy_source(&input, &pool, |original| {
            IsolatedArrayCopy::new(original.array())
                .copy_prepared(
                    &original,
                    &environment,
                    roots,
                    mechanisms,
                    sources.metadata_funding(),
                    &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                )
                .unwrap()
        });
        let source =
            crate::composition::mlx::speculative::sampling::numerical::registered::value_input(
                copy,
                Meaning::Logits,
                sources,
                &environment,
            )
            .unwrap();
        let tensor = crate::MlxTensor::from_array(source.value().array.clone());
        let mut ordinary = RandomState::with_seed(SEED).unwrap();
        let expected = std::array::from_fn::<_, 3, _>(|_| {
            let token = MlxSamplingBackend::sample_processed(
                &tensor,
                1.0,
                Some(&mut ordinary),
                environment.stream(),
            )
            .unwrap();
            let token = token.as_array().evaluated().unwrap().as_slice::<u32>()[0];
            assert!(token < 5);
            (token, ordinary_words(ordinary.as_array()))
        });
        drop((tensor, ordinary));
        let context = SpeculativeExecutionStreams::single(environment.stream())
            .with_original_numerical_sources(sources, &environment)
            .unwrap();
        let mut state = create_key(SEED, context).unwrap();
        for (token, key) in expected {
            assert_eq!(
                sample_stochastic(&DefaultSampler, &source, 1.0, &mut state, context).unwrap(),
                token
            );
            assert_eq!(key_words(&state), key);
        }
        let saved = key_words(&state);
        assert!(sample_stochastic(&DefaultSampler, &source, 0.0, &mut state, context).is_err());
        assert_eq!(key_words(&state), saved);
        pair.request().close().unwrap();
        assert!(sample_stochastic(&DefaultSampler, &source, 1.0, &mut state, context).is_err());
        assert_eq!(key_words(&state), saved);
        assert_eq!(
            source.value().array.evaluated().unwrap().as_slice::<f32>(),
            logits
        );
        (state, expected[2].1)
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    assert_eq!(key_words(&escaped), expected);
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
