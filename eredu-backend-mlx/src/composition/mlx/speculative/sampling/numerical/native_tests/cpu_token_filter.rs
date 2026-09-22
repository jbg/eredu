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
use eredu_runtime::{DefaultSampler, SpeculativeSampler};

#[test]
#[ignore = "requires managed Metal allocator and CPU execution"]
fn original_cpu_policy_composes_mask_temperature_and_escaped_custody() {
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
    let logits = [-2.5f32, 3.75, 3.25, -0.125, 1.5, 50.0, 4.0, -4.0, 7.0, 9.0];
    let (escaped, scaled, scaled_expected) = {
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
                shape: &[1, 2, 5],
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
        let allowed = eredu_core::TokenFilter::Allowed(vec![true, false, true]);
        let history = [1u32, 2];
        let policy =
            <DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(
                &DefaultSampler,
            )
            .unwrap()
            .bind(0.0, history.len())
            .unwrap();
        let mut last = None;
        for forced in [None, Some(2)] {
            let mask = eredu_runtime::generation::TokenMaskPlan::new(
                &allowed,
                source.value().array.shape(),
                forced,
            )
            .unwrap();
            let mut invalid = Vec::with_capacity(mask.elements());
            mask.fill(&mut invalid).unwrap();
            let ordinary = crate::backend::runtime::generation::apply_token_mask(
                &MlxTensor::from_array(source.value().array.clone()),
                &invalid,
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
            for (i, &value) in expected.iter().enumerate() {
                let token = i % 5;
                let kept = (token == 0 || token == 2)
                    && forced.is_none_or(|forced| token == forced as usize);
                assert_eq!(value, if kept { logits[i] } else { f32::NEG_INFINITY });
            }
            let result = NumericalProducer::execute_controlled_policy(
                sources,
                &environment,
                roots,
                mechanisms,
                policy,
                &source,
                &history,
                mask,
            )
            .unwrap();
            let NumericalOutput::Logits(masked) = result else {
                panic!("mask lost logit meaning")
            };
            assert_eq!(
                masked.value().array.evaluated().unwrap().as_slice::<f32>(),
                expected
            );
            assert!(masked
                .value()
                .provenance
                .source()
                .belongs_to_request(pair.request()));
            last = Some(masked);
        }
        assert!(eredu_runtime::generation::TokenMaskPlan::new(
            &allowed,
            source.value().array.shape(),
            Some(4)
        )
        .is_err());
        let scaled =
            <DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::prepared_logit_policy(
                &DefaultSampler,
            )
            .unwrap()
            .bind(0.7, history.len())
            .unwrap();
        let mask = eredu_runtime::generation::TokenMaskPlan::new(
            &allowed,
            source.value().array.shape(),
            None,
        )
        .unwrap();
        let mut invalid = Vec::with_capacity(mask.elements());
        mask.fill(&mut invalid).unwrap();
        let ordinary = crate::backend::runtime::generation::apply_token_mask(
            &MlxTensor::from_array(source.value().array.clone()),
            &invalid,
            environment.stream(),
        )
        .unwrap();
        let ordinary = scaled
            .process::<MlxSamplingBackend>(&ordinary, &history, environment.stream())
            .unwrap();
        let expected = ordinary
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        drop(ordinary);
        for (i, &actual) in expected.iter().enumerate() {
            assert_eq!(
                actual,
                if i % 5 == 0 || i % 5 == 2 {
                    logits[i] * (1.0 / 0.7f32)
                } else {
                    f32::NEG_INFINITY
                }
            );
        }
        let result = NumericalProducer::execute_controlled_policy(
            sources,
            &environment,
            roots,
            mechanisms,
            scaled,
            &source,
            &history,
            mask,
        )
        .unwrap();
        let NumericalOutput::Logits(scaled) = result else {
            panic!("composed policy lost logit meaning")
        };
        assert_eq!(
            scaled
                .value()
                .array
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
            expected
        );
        assert!(scaled
            .value()
            .provenance
            .source()
            .belongs_to_request(pair.request()));
        assert_eq!(
            source.value().array.evaluated().unwrap().as_slice::<f32>(),
            logits
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        (last.unwrap(), scaled, expected)
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    drop((target, draft));
    assert_eq!(
        escaped.value().array.evaluated().unwrap().as_slice::<f32>(),
        &[
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            3.25,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            -4.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY
        ]
    );
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
    drop((escaped, scaled));
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
