use super::*;
use eredu_core::{TextGenerationConfig, TokenFilter, TokenFilterController};
use eredu_runtime::{MirostatV2Sampler, SamplingBackend};

#[derive(Clone, Default)]
struct TestConstraint {
    committed: Vec<u32>,
}

impl TokenFilterController for TestConstraint {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.filter_at(&self.committed)
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert_ne!(token, 1);
        self.committed.push(token);
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

impl SpeculativeTokenFilterController for TestConstraint {
    type PreparedGrammar = eredu_core::speculative::NoPreparedGrammar;
    fn filter_at(&self, history: &[u32]) -> Result<TokenFilter, Self::Error> {
        assert!(history.starts_with(&self.committed));
        Ok(TokenFilter::allowed(vec![true, false, true, true]).unwrap())
    }

    fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn native_preparation_speculative_sampling_rejects_unintegrated_policy() {
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let memory = NativeMemoryOwner::acquire(&pool).unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.7),
            ..Default::default()
        },
    )
    .unwrap();
    let chunk = eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(2),
        ..Default::default()
    };
    let generation = TextGenerationConfig::new(sampling).with_inference_policy(chunk);
    validate_speculative_prefill_policy(generation.clone(), true).unwrap();
    let error = validate_speculative_prefill_policy(generation.clone(), false).unwrap_err();
    let Error::Other(source) = error else {
        panic!("typed policy rejection")
    };
    assert!(matches!(
        source.downcast_ref::<eredu_core::CapabilityError>(),
        Some(eredu_core::CapabilityError::InvalidConfiguration {
            field: "speculative_inference_policy",
            ..
        })
    ));
    let managed = eredu_core::TextInferencePolicy {
        memory_limits: eredu_core::MemoryLimitDeclarations::new([(
            "host".into(),
            eredu_core::MemoryLimit::Finite(u64::MAX),
        )]),
        ..Default::default()
    };
    for independent in [false, true] {
        let generation = TextGenerationConfig::new(sampling).with_inference_policy(managed.clone());
        assert!(validate_speculative_prefill_policy(generation.clone(), independent).is_err());
        let error = MlxSpeculativeSession::prepare_mlx_speculative_sampling(
            generation,
            TestConstraint::default(),
            &memory,
        )
        .err()
        .expect("unknown memory bound rejects before sampler construction");
        let Error::Other(source) = error else {
            panic!("typed memory rejection")
        };
        assert_eq!(
            source.downcast_ref::<eredu_runtime::working_memory::WorkingMemoryError>(),
            Some(&eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
        );
    }
}

#[test]
fn prepared_mirostat_preserves_seed_penalties_constraints_and_forked_state() {
    let pool = crate::memory_fixture::ledger(0, 0).unwrap();
    let memory = NativeMemoryOwner::acquire(&pool).unwrap();
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let resolved = eredu_core::generation::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            temperature: Some(0.8),
            top_k: Some(1),
            top_p: Some(0.1),
            min_p: Some(0.99),
            repetition_penalty: Some(1.2),
            repeat_last_n: Some(2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.4),
            ..Default::default()
        },
    )
    .unwrap();
    let generation = TextGenerationConfig::new(resolved)
        .with_seed(73)
        .with_mirostat_v2(4.0, 0.2)
        .unwrap();
    let (key, mut sampler) = MlxSpeculativeSession::prepare_mlx_speculative_sampling(
        generation,
        TestConstraint::default(),
        &memory,
    )
    .unwrap();
    assert_eq!(
        key.unwrap()
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<u32>(),
        safemlx::random::key(73)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<u32>()
    );
    assert!(
        !SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(&sampler)
    );
    let (_, standard) = MlxSpeculativeSession::prepare_mlx_speculative_sampling(
        TextGenerationConfig::new(resolved),
        TestConstraint::default(),
        &memory,
    )
    .unwrap();
    assert!(
        SpeculativeSampler::<MlxSamplingBackend>::supports_exact_optimistic_promotion(&standard)
    );

    let mut reference = ConstrainedSampler::new(
        MirostatV2Sampler::new(4.0, 0.2)
            .unwrap()
            .penalties(1.2, 2, 0.3, 0.4),
        TestConstraint::default(),
    );
    let logits = MlxTensor::from_array(Array::from_slice(&[1.0f32, 100.0, 0.8, -0.5], &[1, 4]));
    let mut history = Vec::new();
    let mut expected_mu = 8.0;
    for token in [0, 2, 0] {
        let processed = SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut sampler,
            &logits,
            0.8,
            &history,
            &stream,
        )
        .unwrap();
        let expected = SpeculativeSampler::<MlxSamplingBackend>::process_logits(
            &mut reference,
            &logits,
            0.8,
            &history,
            &stream,
        )
        .unwrap();
        let values = processed.as_array().evaluated().unwrap();
        assert_eq!(
            values.as_slice::<f32>(),
            expected.as_array().evaluated().unwrap().as_slice::<f32>()
        );
        // Mirostat replaces top-k/p/min-p while the vocabulary mask is retained.
        assert_eq!(
            values
                .as_slice::<f32>()
                .iter()
                .filter(|v| v.is_finite())
                .count(),
            3
        );
        assert_eq!(values.as_slice::<f32>()[1], f32::NEG_INFINITY);
        let probability =
            MlxSamplingBackend::token_probability(&processed, token, &stream).unwrap();
        expected_mu -= 0.2 * (-probability.log2() - 4.0);
        SpeculativeSampler::<MlxSamplingBackend>::commit_token(
            &mut sampler,
            &processed,
            token,
            &stream,
        )
        .unwrap();
        history.push(token);
        let MlxTextSampler::MirostatV2(policy) = sampler.policy() else {
            panic!("prepared sampler must retain Mirostat V2");
        };
        assert_eq!(policy.generated_tokens(), history);
        assert_eq!(sampler.controller().committed, history);
        assert!((policy.mu() - expected_mu).abs() < 1e-6);
        SpeculativeSampler::<MlxSamplingBackend>::commit_token(
            &mut reference,
            &expected,
            token,
            &stream,
        )
        .unwrap();
    }

    let mut fork = sampler.clone();
    let processed = SpeculativeSampler::<MlxSamplingBackend>::process_logits(
        &mut fork, &logits, 0.8, &history, &stream,
    )
    .unwrap();
    SpeculativeSampler::<MlxSamplingBackend>::commit_token(&mut fork, &processed, 3, &stream)
        .unwrap();
    assert_eq!(fork.controller().committed, [0, 2, 0, 3]);
    assert_eq!(sampler.controller().committed, history);
    let MlxTextSampler::MirostatV2(policy) = sampler.policy() else {
        unreachable!()
    };
    assert_eq!(policy.generated_tokens(), history);
    assert!((policy.mu() - expected_mu).abs() < 1e-6);
}

#[test]
fn lane_proposal_width_cannot_exceed_selected_realization() {
    let config = SpeculativeConfig {
        max_draft_tokens: 5,
        ..SpeculativeConfig::default()
    };
    let error = validate_lane_proposal_capacity(&config, 4).unwrap_err();
    assert!(error.to_string().contains("admits at most 4"));
    validate_lane_proposal_capacity(&config, 5).unwrap();
}

#[test]
fn fused_rows_are_selected_by_backend_mechanisms_without_family_policy() {
    let device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device);
    let value = MlxTensor::from_array(Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
    ));
    assert_eq!(
        MlxEmbeddedPredictionMechanisms::sequence_len(&value).unwrap(),
        2
    );
    let row = MlxEmbeddedPredictionMechanisms::fused_logits_row(
        &value,
        1,
        SpeculativeExecutionStreams::single(&stream),
    )
    .unwrap();
    let IndependentLogits::Ordinary(row) = row else {
        panic!("ordinary fused row");
    };
    assert_eq!(row.evaluated().unwrap().as_slice::<f32>(), &[4.0, 5.0, 6.0]);
}
