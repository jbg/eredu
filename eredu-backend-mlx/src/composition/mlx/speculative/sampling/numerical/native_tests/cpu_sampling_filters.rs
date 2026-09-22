//! Public vocabulary-mask producer on the actual admitted CPU numerical path.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::MlxPreparedInputMaterializer;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use eredu_runtime::{GenerationSampler, SpeculativeSampler};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_filters_preserve_shared_policy_rounding_and_output_custody() {
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
    let logits: Vec<f32> = (0..65)
        .map(|i| ((i * 17) % 41) as f32 * 0.125 - 2.5)
        .collect();
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
                shape: &[1, 1, 65],
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
        let source = registered::value_input(copy, Meaning::Logits, sources, &environment).unwrap();
        let mut final_output = None;
        for (top_k, top_p, min_p, temperature) in [
            (1, 1.0, 0.0, 0.0),
            (7, 1.0, 0.0, 0.0),
            (0, 0.83, 0.0, 0.0),
            (0, 1.0, 0.15, 0.0),
            (40, 0.95, 0.05, 0.7),
        ] {
            let sampler = GenerationSampler::new()
                .top_k(top_k)
                .top_p(top_p)
                .min_p(min_p);
            let policy=<GenerationSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(&sampler)
                .unwrap().bind(temperature,0).unwrap();
            let ordinary = policy
                .process::<MlxSamplingBackend>(
                    &MlxTensor::from_array(source.value().array.clone()),
                    &[],
                    environment.stream(),
                )
                .unwrap();
            let expected = ordinary
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            drop(ordinary);
            assert!(expected.iter().any(|x| x.is_finite()));
            assert!(expected.iter().any(|x| x.is_infinite()));
            if top_k > 0 && top_p == 1.0 && min_p == 0.0 {
                let mut sorted = logits.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let threshold = sorted[top_k as usize - 1];
                for (actual, original) in expected.iter().zip(&logits) {
                    assert_eq!(
                        *actual,
                        if *original < threshold {
                            f32::NEG_INFINITY
                        } else {
                            *original
                        }
                    );
                }
            }
            let mask = eredu_runtime::generation::TokenMaskPlan::new(
                &eredu_core::TokenFilter::All,
                source.value().array.shape(),
                None,
            )
            .unwrap();
            let output = NumericalProducer::execute_controlled_policy(
                sources,
                &environment,
                roots,
                mechanisms,
                policy,
                &source,
                &[],
                mask,
            )
            .unwrap();
            let NumericalOutput::Logits(output) = output else {
                panic!("filtered output lost logit meaning")
            };
            assert_eq!(
                output
                    .value()
                    .array
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                expected
            );
            assert!(output
                .value()
                .provenance
                .source()
                .belongs_to_request(pair.request()));
            final_output = Some((output, expected));
        }
        assert_eq!(
            source.value().array.evaluated().unwrap().as_slice::<f32>(),
            logits
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        final_output.unwrap()
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    drop((target, draft));
    assert_eq!(
        escaped
            .value()
            .array
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        expected
    );
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
