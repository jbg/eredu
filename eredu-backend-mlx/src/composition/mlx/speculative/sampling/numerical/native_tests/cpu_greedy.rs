//! Actual shared CPU greedy worker from an admitted registered F32 source.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::MlxPreparedInputMaterializer;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use eredu_runtime::{DefaultSampler, SpeculativeSampler};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_greedy_matches_first_tie_and_retires_completed_phase() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    exercise(false)
}
#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_policy_preserves_identity_scales_and_commits_on_draft() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    exercise(true)
}
fn exercise(policy_consumer: bool) {
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
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: 0.0,
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
    let logits = [-2.5f32, 3.75, 3.75, -0.125, 1.5];
    {
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
        let choice =
            <DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_greedy_policy(
                &DefaultSampler,
            )
            .unwrap()
            .greedy(0.0)
            .unwrap();
        let tensor = MlxTensor::from_array(source.value().array.clone());
        let ordinary = choice
            .construct::<MlxSamplingBackend>(&tensor, environment.stream())
            .unwrap();
        let expected = ordinary.as_array().evaluated().unwrap().as_slice::<u32>()[0];
        assert_eq!(expected, 1);
        drop(ordinary);
        drop(tensor);
        if policy_consumer {
            let context = SpeculativeExecutionStreams::single(environment.stream())
                .with_original_numerical_sources(sources, &environment)
                .unwrap();
            let processed = process_policy_at(
                &DefaultSampler,
                &source,
                0.0,
                &[],
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
            assert!(
                Rc::ptr_eq(processed.0.as_ref().unwrap(), source.0.as_ref().unwrap()),
                "the shared empty trace returns the same completed source owner"
            );
            let actual = sample_greedy_at(
                &DefaultSampler,
                &processed,
                0.0,
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
            assert_eq!(actual, expected);
            commit_without_mutation(
                &DefaultSampler,
                &processed,
                actual,
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
            assert!(
                validate_commit_value(&processed, 5, SamplingPlacement::Draft, context).is_err(),
                "the same token domain is enforced on CPU"
            );
            let temperature = 0.7f32;
            let policy =
                <DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(
                    &DefaultSampler,
                )
                .unwrap()
                .bind(temperature, 0)
                .unwrap();
            let ordinary = policy
                .process::<MlxSamplingBackend>(
                    &MlxTensor::from_array(source.value().array.clone()),
                    &[],
                    environment.stream(),
                )
                .unwrap();
            let scaled_expected = ordinary
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            drop(ordinary);
            for (&actual, &input) in scaled_expected.iter().zip(&logits) {
                assert_eq!(actual, input * (1.0 / temperature));
            }
            let scaled = process_policy_at(
                &DefaultSampler,
                &source,
                temperature,
                &[],
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
            assert!(!Rc::ptr_eq(
                scaled.0.as_ref().unwrap(),
                source.0.as_ref().unwrap()
            ));
            assert_eq!(
                scaled
                    .value()
                    .array
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                scaled_expected
            );
            assert!(scaled
                .value()
                .provenance
                .source()
                .belongs_to_request(pair.request()));
            let actual = sample_greedy_at(
                &DefaultSampler,
                &scaled,
                0.0,
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
            assert_eq!(actual, expected);
            commit_without_mutation(
                &DefaultSampler,
                &scaled,
                actual,
                SamplingPlacement::Draft,
                context,
            )
            .unwrap();
        } else {
            let output = NumericalProducer::execute(
                sources,
                &environment,
                roots,
                mechanisms,
                program::SpeculativeNumericalKind::Greedy(choice),
                &source,
                None,
            )
            .unwrap();
            let NumericalOutput::Token(actual) = output else {
                panic!("greedy lost scalar token meaning")
            };
            assert_eq!(actual, expected);
        }
        assert!(source
            .value()
            .provenance
            .source()
            .belongs_to_request(pair.request()));
        assert_eq!(
            source.value().array.evaluated().unwrap().as_slice::<f32>(),
            logits
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
    drop((target, draft));
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
