//! Categorical consumes the actual final prefill row, never raw Array + role.
use super::*;
use crate::backend::runtime::generation::MlxSamplingBackend;
use crate::composition::mlx::{
    MlxPreparedInputMaterializer,
    speculative::{
        autoregressive::MlxAutoregressiveMechanisms as Mechanisms,
        sampling::logits::IndependentLogits,
    },
};
use eredu_core::{GenerationCancellationToken, SpeculativePrefillOutcome};
use eredu_runtime::{
    DefaultSampler, SamplingBackend,
    generation::SpeculativeSampler,
    speculative::autoregressive::{AutoregressiveMechanisms, AutoregressivePass},
};
const TEMPERATURE: f32 = 0.7;

#[test]
#[ignore = "requires native Metal execution"]
fn original_categorical_uses_funded_target_prefill_and_preserves_key_advancement() {
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
    let (raw_logits, processed_logits, actual_tokens, actual_state) = {
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
        // A second real issuance deliberately shares the same schedule, exact
        // loaded sources and host metadata account. It has a different request
        // ticket and must never borrow this request's state snapshot.
        let foreign = AutoregressiveSourcePair::prepare_funded(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
            pair.metadata_funding().clone(),
        )
        .unwrap();
        assert_eq!(pair.schedule_identity(), foreign.schedule_identity());
        assert!(
            !pair
                .request()
                .source_identity()
                .belongs_to_request(foreign.request())
        );
        let foreign_context = SpeculativeExecutionStreams::single(backend.stream())
            .with_original_sources(&foreign, &environment)
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
        // The actual completed two-span state is copied by the shared checkpoint
        // worker. Foreign restore refuses before spending H or admitting native
        // work; the same snapshot still restores through its original request.
        let saved = Mechanisms::checkpoint(&cache).unwrap();
        let saved_estimate = Mechanisms::estimate(&saved);
        let before_refusal = pool.used_bytes().unwrap();
        let refused = Mechanisms::restore(&saved, foreign_context)
            .err()
            .expect("same-header foreign request must not restore state");
        assert!(matches!(
            refused,
            crate::backend::error::Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch
            )
        ));
        assert_eq!(pool.used_bytes().unwrap(), before_refusal);
        assert_eq!(Mechanisms::estimate(&saved), saved_estimate);
        let restored = Mechanisms::restore(&saved, context).unwrap();
        assert_eq!(Mechanisms::estimate_state(&restored), saved_estimate);
        drop((restored, saved, foreign));
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
        eprintln!("original categorical: prepared policy");
        let processed =
            process_policy(&DefaultSampler, &logits, TEMPERATURE, &[], context).unwrap();
        let processed_values = processed
            .value()
            .array
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let mut state = create_key(SEED, context).unwrap();
        eprintln!("original categorical: two real-key draws");
        let tokens = [
            sample_stochastic(
                &DefaultSampler,
                &processed,
                TEMPERATURE,
                &mut state,
                context,
            )
            .unwrap(),
            sample_stochastic(
                &DefaultSampler,
                &processed,
                TEMPERATURE,
                &mut state,
                context,
            )
            .unwrap(),
        ];
        assert!(tokens.iter().all(|token| *token < 64));
        let final_key = key_words(&state);
        // This policy cannot describe a stochastic draw at zero temperature.
        // It must refuse without invoking another key-advancement transaction.
        let refused =
            sample_stochastic(&DefaultSampler, &processed, 0.0, &mut state, context).unwrap_err();
        assert_eq!(key_words(&state), final_key);
        drop(refused);
        pair.request().close().unwrap();
        drop((logits, processed, cache, state, pair));
        (raw, processed_values, tokens, final_key)
    };
    settle(&pool, loaded);

    // Use the exact observed model row with the unchanged ordinary policy and
    // random worker after original request retirement; no managed raw escape.
    let stream = backend.stream();
    let raw = crate::MlxTensor::from_array(Array::from_slice(&raw_logits, &[1, 64]));
    let processed = <DefaultSampler as SpeculativeSampler<MlxSamplingBackend>>::process_logits(
        &mut DefaultSampler,
        &raw,
        TEMPERATURE,
        &[],
        stream,
    )
    .unwrap();
    assert_eq!(
        processed.as_array().evaluated().unwrap().as_slice::<f32>(),
        processed_logits
    );
    let mut reference = RandomState::with_seed(SEED).unwrap();
    for actual in actual_tokens {
        let expected = MlxSamplingBackend::sample_processed(
            &processed,
            TEMPERATURE,
            Some(&mut reference),
            stream,
        )
        .unwrap();
        assert_eq!(
            expected.as_array().evaluated().unwrap().as_slice::<u32>(),
            &[actual]
        );
    }
    assert_eq!(ordinary_words(reference.as_array()), actual_state);
    drop((
        raw,
        processed,
        reference,
        target,
        draft,
        target_config,
        draft_config,
        selected,
    ));
    settle(&pool, initial);
}
