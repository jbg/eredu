//! Adaptive state consumes an actual original prefill row and actual PRNG phases.
use super::*;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::{
    MlxPreparedInputMaterializer,
    speculative::{
        autoregressive::MlxAutoregressiveMechanisms as Mechanisms,
        sampling::logits::IndependentLogits,
    },
};
use eredu_core::{GenerationCancellationToken, SpeculativePrefillOutcome, speculative::SamplingPlacement};
use eredu_runtime::{
    ConfiguredTextSampler, MirostatV2Sampler, SamplingBackend,
    generation::SpeculativeSampler,
    speculative::autoregressive::{AutoregressiveMechanisms, AutoregressivePass},
};
const TEMPERATURE: f32 = 0.7;

#[test]
#[ignore = "requires native Metal execution"]
fn original_mirostat_preserves_mu_history_snapshot_and_native_retirement() {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let initial = pool.used_bytes().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let mut target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.used_bytes().unwrap();
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: TEMPERATURE,
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
    let (raw_logits, records, actual_state) = {
        let mut invocation = None;
        schedule
            .domains()
            .iter()
            .find(|domain| domain.pass() == AutoregressivePass::TargetPrefill)
            .unwrap()
            .visit(|frontier, value| {
                assert_eq!(frontier, 0);
                assert_eq!(value.positions(), 2);
                invocation = Some(value);
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap();
        // Both host I and native/full-control B are constructed by the same
        // production preparation used by original text ingress. No raw array is
        // adopted into the registry or relabeled as a model-role grant.
        let ids = [1u32, 3];
        let parts = [eredu_runtime::input::host::HostInputPart {
            modality: eredu_core::InputModality::Text,
            kind: eredu_core::InputPayloadKind::TokenIds,
            payload: eredu_runtime::input::host::HostTensorView {
                shape: &[1, 2],
                values: eredu_runtime::input::host::HostTensorValues::U32(&ids),
            },
            metadata: &[],
            extents: &[],
        }];
        let source = pool
            .compile_prepared_host_input(
                eredu_runtime::input::host::PreparedHostInputPlan::prepare(&parts).unwrap(),
            )
            .unwrap();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let plan = materializer.text_input_plan(&source).unwrap();
        assert!(plan.required_bytes().unwrap() > 0);
        let prompt = plan
            .materialize(&pool)
            .unwrap()
            .into_prompt(NonZeroU64::new(1));
        drop(source); // the completed native/parts packet independently retains I/B
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        let cancellation = GenerationCancellationToken::new();
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
        )
        .unwrap();
        let environment = backend.original_copy_environment().unwrap();
        let context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&pair, &environment)
            .unwrap();
        let mut cursor = schedule.into_cursor();
        let claim = cursor.claim(0, invocation.unwrap()).unwrap();
        eprintln!("original categorical: funded target prefill");
        let (completed, cache) = target
            .with_model_operation_funded(pair.metadata_funding().clone(), |model| {
                let mut cache =
                    Mechanisms::empty(model, AutoregressivePass::TargetPrefill, context)?;
                let completed = Mechanisms::with_invocation(
                    model,
                    &mut cache,
                    Some(&prompt),
                    claim,
                    context,
                    |model, cache| {
                        Mechanisms::prefill(
                            model,
                            &prompt,
                            cache,
                            AutoregressivePass::TargetPrefill,
                            &cancellation,
                            context,
                        )
                    },
                )?;
                Ok((completed, cache))
            })
            .unwrap();
        drop(prompt); // subsequent numerical phases use only completed model output
        let SpeculativePrefillOutcome::Complete(completed) = completed else {
            panic!("uncancelled actual prefill did not complete");
        };
        assert_eq!(completed.evaluated_tokens, 2);
        let Some(IndependentLogits::Original(logits)) = completed.logits else {
            panic!("actual completed original prefill row required");
        };
        let raw = logits
            .value()
            .array
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(raw.len(), 64);
        assert!(raw.iter().all(|value| value.is_finite()));
        assert!(raw.iter().any(|value| *value != 0.0));
        let policy = ConfiguredTextSampler::MirostatV2(
            MirostatV2Sampler::new(3.5, 0.2)
                .unwrap()
                .penalties(1.1, 32, 0.2, 0.1),
        );
        let mut sampler: MlxSpeculativeSampling<ConfiguredTextSampler> =
            MlxSpeculativeSampling::prepare(policy, context, None).unwrap();
        let mut state = create_key(SEED, context).unwrap();
        let mut records = Vec::new();
        for step in 0..3 {
            let history = adaptive_state(sampler.inner.source())
                .generated_tokens()
                .to_vec();
            let processed = process_policy(
                sampler.inner.source(),
                &logits,
                TEMPERATURE,
                &history,
                context,
            )
            .unwrap_or_else(|error| panic!("adaptive processing {step}: {error}"));
            let values = processed
                .value()
                .array
                .evaluated()
                .unwrap()
                .as_slice::<f32>()
                .to_vec();
            let token = sample_stochastic(
                sampler.inner.source(),
                &processed,
                TEMPERATURE,
                &mut state,
                context,
            )
            .unwrap_or_else(|error| panic!("adaptive choice {step}: {error}"));
            let bytes =
                sampler
                    .original_snapshot_metadata(None, None)
                    .unwrap()
                    .checked_add(
                        eredu_core::HostPreparationAuthority::retention_bytes::<
                            HostMetadataFunding,
                        >()
                        .unwrap(),
                    )
                    .unwrap();
            pair.metadata_funding().reserve_metadata(bytes).unwrap();
            let (snapshot, _, _) = sampler
                .copy_snapshot(
                    None,
                    None,
                    eredu_core::HostPreparationAuthority::retain(pair.metadata_funding().clone()),
                )
                .unwrap();
            let mu_before = adaptive_state(snapshot.inner.source()).mu().to_bits();
            sampler
                .commit_original(&processed, token, SamplingPlacement::Draft, context)
                .unwrap_or_else(|error| panic!("adaptive commit {step}: {error}"));
            assert_eq!(
                adaptive_state(snapshot.inner.source()).mu().to_bits(),
                mu_before
            );
            assert_eq!(
                adaptive_state(snapshot.inner.source()).generated_tokens(),
                history
            );
            assert_eq!(
                adaptive_state(sampler.inner.source())
                    .generated_tokens()
                    .last(),
                Some(&token)
            );
            records.push((
                values,
                token,
                adaptive_state(sampler.inner.source()).mu().to_bits(),
            ));
        }
        let final_key = key_words(&state);
        pair.request().close().unwrap();
        drop((sampler, logits, cache, state, pair));
        (raw, records, final_key)
    };
    settle(&pool, loaded);
    // Run the same ordinary worker after original request retirement.
    let stream = backend.stream();
    let raw = crate::MlxTensor::from_array(Array::from_slice(&raw_logits, &[1, 64]));
    let mut policy = ConfiguredTextSampler::MirostatV2(
        MirostatV2Sampler::new(3.5, 0.2)
            .unwrap()
            .penalties(1.1, 32, 0.2, 0.1),
    );
    let mut random = RandomState::with_seed(SEED).unwrap();
    for (values, actual_token, actual_mu) in records {
        let history = adaptive_state(&policy).generated_tokens().to_vec();
        let processed =
            <ConfiguredTextSampler as SpeculativeSampler<MlxSamplingBackend>>::process_logits(
                &mut policy,
                &raw,
                TEMPERATURE,
                &history,
                stream,
            )
            .unwrap();
        assert_eq!(
            processed.as_array().evaluated().unwrap().as_slice::<f32>(),
            values
        );
        let token = MlxSamplingBackend::sample_processed(
            &processed,
            TEMPERATURE,
            Some(&mut random),
            stream,
        )
        .unwrap();
        let expected = MlxSamplingBackend::token_id(&token, stream).unwrap();
        assert_eq!(expected, actual_token);
        <ConfiguredTextSampler as SpeculativeSampler<MlxSamplingBackend>>::commit_token(
            &mut policy,
            &processed,
            expected,
            stream,
        )
        .unwrap();
        assert_eq!(adaptive_state(&policy).mu().to_bits(), actual_mu);
    }
    assert_eq!(ordinary_words(random.as_array()), actual_state);
    drop((
        raw,
        policy,
        random,
        target,
        draft,
        target_config,
        draft_config,
        selected,
    ));
    settle(&pool, initial);
}
fn adaptive_state(policy: &ConfiguredTextSampler) -> &MirostatV2Sampler {
    let ConfiguredTextSampler::MirostatV2(policy) = policy else {
        panic!("adaptive policy")
    };
    policy
}
