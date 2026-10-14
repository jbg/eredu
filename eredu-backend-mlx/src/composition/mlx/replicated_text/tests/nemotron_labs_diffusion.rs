mod adapter;
mod causal;
mod table_diffusion;
mod independent_modes;
mod ring;
mod vlm_reference;
mod vlm_modes;
use super::*;
use crate::composition::mlx::{
    MlxBackend, speculative::autoregressive::MlxAutoregressiveMechanisms,
};
use eredu_core::{
    Completion, ControlledTextGeneration, ModelRuntime, SpeculativeExecutor,
    SpeculativeProposalStep, TokenFilterController,
};
use eredu_runtime::speculative::shared_backbone::{SharedBackboneExecutor, SharedBackboneInput};

fn config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"nemotron_labs_diffusion", "hidden_size":6, "head_dim":4,
        "num_attention_heads":2, "num_key_value_heads":1, "num_hidden_layers":2,
        "intermediate_size":7, "vocab_size":11, "rms_norm_eps":0.00001,
        "block_size":3, "mask_token_id":0, "hidden_act":"silu", "dlm_paradigm":"bidirectional",
        "tie_word_embeddings":false, "attention_bias":false, "mlp_bias":false,
        "max_position_embeddings":64, "sliding_window":null,
        "rope_parameters":{"rope_type":"yarn","factor":2.0,"rope_theta":10000.0,
            "original_max_position_embeddings":4,"beta_fast":32.0,"beta_slow":1.0,
            "mscale":1.0,"mscale_all_dim":1.0,"llama_4_scaling_beta":0.1}
    })
}

fn frontier<'a>(
    executor: &SharedBackboneExecutor<'a, MlxAutoregressiveMechanisms>,
    cache: &<SharedBackboneExecutor<'a, MlxAutoregressiveMechanisms> as SpeculativeExecutor>::Cache,
    seed: &<SharedBackboneExecutor<'a, MlxAutoregressiveMechanisms> as SpeculativeExecutor>::TargetState,
) -> u64 {
    executor
        .continuation_memory_observation(cache, seed, 0)
        .unwrap()
        .unwrap()
        .target
        .current_positions
}

#[derive(Clone)]
struct ExcludeMask;
impl TokenFilterController for ExcludeMask {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::allowed((0..11).map(|id| id != 0).collect()).unwrap())
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        assert_ne!(token, 0);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
impl eredu_core::SpeculativeTokenFilterController for ExcludeMask {
    fn filter_at(&self, _: &[u32]) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::allowed((0..11).map(|id| id != 0).collect()).unwrap())
    }
    fn control_snapshot_bytes(&self) -> Option<u64> {
        Some(0)
    }
    fn prefix_is_complete(&self, _: &[u32]) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn prepared_shared_request_table_handoffs_match_ar_and_restore_retained_scores() {
    use crate::composition::mlx::speculative::{
        MlxSpeculativeSampling, SpeculativeExecutionStreams,
    };
    use eredu_core::decoding::{DecodingMode, SpeculationOptions};
    use eredu_core::{
        GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
        SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
    };
    use eredu_runtime::ConstrainedSampler;
    let root = tiny_heterogeneous_artifact(config());
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let model = eredu_core::load_model(
            &backend,
            root.path(),
            crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                    eredu_runtime::WeightResidency::with_layers(policy.layers()),
                ),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(6),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let source_sampler = crate::composition::mlx::session::MlxTextSampler::from_config(
            &eredu_core::TextGenerationConfig::new(sampling),
        )
        .unwrap();
        let baseline = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        runtime.reset().unwrap();
        let prompt = <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
            runtime.backend(),
            vec![1, 4, 2],
        )
        .unwrap();
        runtime
            .session_mut()
            .with_model_operation(|model| {
                let context = SpeculativeExecutionStreams::single(&stream);
                let options = SpeculationOptions::default()
                    .with_proposal_capacity(std::num::NonZeroUsize::new(1).unwrap());
                let draft = model
                    .erased()
                    .capability_estimate()
                    .shared_backbone_draft()
                    .unwrap()
                    .prepare(options, 0.0)
                    .unwrap();
                let mut executor = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                    model.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                );
                let mut cache = executor.new_cache(context)?;
                let mut delivered = Vec::new();
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(
                        source_sampler,
                        ExcludeMask,
                    )),
                    eredu_core::GenerationSequence::new(6, []),
                    SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|tokens| {
                        delivered.extend_from_slice(tokens);
                        Ok(())
                    }),
                    Default::default(),
                );
                let mut table = GenerationRequestTable::new(
                    Default::default(),
                    eredu_core::SpeculativeExecutionTopology::Single,
                )
                .unwrap();
                let id = table
                    .submit(
                        &mut executor,
                        &mut cache,
                        SharedBackboneInput::new(
                            prompt,
                            draft,
                            crate::backend::random::RandomState::with_seed(17)?,
                            crate::backend::random::RandomState::with_seed(29)?,
                        ),
                        GenerationRequestConfig {
                            max_tokens: 6,
                            decoding_mode: DecodingMode::SelfSpeculation(options),
                            ..Default::default()
                        },
                        output,
                        SpeculativeRandomness::new(None, None),
                        false,
                        context,
                    )
                    .map_err(|error| {
                        Error::Exception(safemlx::error::Exception::from_source(error))
                    })?;
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::Autoregressive)
                    .unwrap();
                table.step(&mut executor, false, context).unwrap();
                let request = table.request(id).unwrap();
                assert_eq!(
                    request
                        .decoding_state()
                        .acknowledgement
                        .unwrap()
                        .committed_position,
                    1
                );
                assert_eq!(
                    request
                        .continuation_memory_observation(&executor, 0)
                        .unwrap()
                        .unwrap()
                        .target
                        .current_positions,
                    4
                );
                assert!(request.pending_committed_tokens().is_empty());
                table.step(&mut executor, false, context).unwrap();
                assert_eq!(
                    table
                        .request(id)
                        .unwrap()
                        .continuation_memory_observation(&executor, 0)
                        .unwrap()
                        .unwrap()
                        .target
                        .current_positions,
                    4
                );
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::SelfSpeculation(
                        options.with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap()),
                    ))
                    .unwrap();
                table.step(&mut executor, false, context).unwrap();
                let request = table.request(id).unwrap();
                assert_eq!(
                    request
                        .decoding_state()
                        .acknowledgement
                        .unwrap()
                        .committed_position,
                    2
                );
                let limits = eredu_core::execution_control::SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 0,
                    retained_bytes: 64 << 20,
                    cumulative_copy_bytes: 128 << 20,
                };
                let budget = eredu_runtime::execution_control::SnapshotBudget::new(limits);
                let estimate = request.control_snapshot_estimate(&executor).unwrap();
                let _reservation = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Snapshot,
                        Some(estimate),
                    )
                    .unwrap();
                let saved = request.control_snapshot(&executor, context).unwrap();
                table.run(&mut executor, false, context).unwrap();
                assert_eq!(table.request(id).unwrap().sequence().tokens(), baseline);
                let _restore = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Restore,
                        Some(estimate),
                    )
                    .unwrap();
                table
                    .request_mut(id)
                    .unwrap()
                    .restore_control_snapshot(&mut executor, &saved, context)
                    .unwrap();
                table.run(&mut executor, false, context).unwrap();
                assert_eq!(table.request(id).unwrap().sequence().tokens(), baseline);
                table.finish().unwrap();
                assert_eq!(&delivered[..baseline.len()], baseline);
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn loaded_mode_changes_append_pending_suffix_once_and_preserve_next_causal_logits() {
    use eredu_core::{
        TextAdvancement, decoding::DecodingMode, execution_control::GenerationControlHandle,
    };
    let root = tiny_heterogeneous_artifact(config());
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let request = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::with_layers(policy.layers()),
            ),
        );
        let model = eredu_core::load_model(&backend, root.path(), request).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(6),
                temperature: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let handle = GenerationControlHandle::default();
        let mut generation = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap();
        generation.bind_control_handle(&handle).unwrap();
        let mut prefix = vec![1, 4, 2];
        for _ in 0..2 {
            prefix.push(generation.next().unwrap().unwrap().token_id());
        }
        handle
            .request_decoding_mode(DecodingMode::Diffusion { policy: None })
            .unwrap();
        let TextAdvancement::ModeChanged(ack) = generation.advance_operation().unwrap().unwrap()
        else {
            panic!("AR handoff is a settled boundary")
        };
        assert_eq!(ack.committed_position, 2);
        prefix.push(generation.next().unwrap().unwrap().token_id());
        handle
            .request_decoding_mode(DecodingMode::Autoregressive)
            .unwrap();
        loop {
            match generation.advance_operation().unwrap().unwrap() {
                TextAdvancement::Token(token) => prefix.push(token.token_id()),
                TextAdvancement::Progress(_) => {}
                TextAdvancement::ModeChanged(ack) => {
                    assert_eq!(
                        ack.committed_position, 5,
                        "transition waits for all three block commitments"
                    );
                    break;
                }
            }
        }
        let next = generation.next().unwrap().unwrap().token_id();
        drop(generation);
        assert_eq!(
            runtime
                .session_mut()
                .with_model_operation(|model| Ok(model.erased().forecast_state_offset()?))
                .unwrap(),
            Some(prefix.len() as i32)
        );
        runtime.reset().unwrap();
        let expected = ControlledTextGeneration::new(
            &mut runtime,
            prefix,
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .token_id();
        assert_eq!(
            next, expected,
            "settlement preserves the next conditional distribution"
        );
    }
}

#[test]
fn physical_shared_speculation_matches_greedy_ar_and_all_mode_handoffs_settle() {
    use eredu_core::{
        TextAdvancement,
        decoding::{DecodingMode, SpeculationOptionsRequest},
        execution_control::GenerationControlHandle,
    };
    let root = tiny_heterogeneous_artifact(config());
    let modes = [
        DecodingMode::Autoregressive,
        DecodingMode::Diffusion { policy: None },
        DecodingMode::SelfSpeculation(
            SpeculationOptionsRequest {
                max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                ..Default::default()
            }
            .try_into()
            .unwrap(),
        ),
    ];
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(10),
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap();
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let request = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::with_layers(policy.layers()),
            ),
        );
        let model = eredu_core::load_model(&backend, root.path(), request).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let expected = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        runtime.reset().unwrap();
        let actual = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling).with_decoding_mode(Some(modes[2])),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "{policy:?}: self speculation preserves greedy target output"
        );
        for (start, initial) in modes.into_iter().enumerate() {
            for (end, requested) in modes.into_iter().enumerate() {
                if start == end {
                    continue;
                }
                runtime.reset().unwrap();
                let handle = GenerationControlHandle::default();
                let mut generation = ControlledTextGeneration::new(
                    &mut runtime,
                    vec![1, 4, 2],
                    eredu_core::TextGenerationConfig::new(sampling)
                        .with_decoding_mode(Some(initial)),
                    ExcludeMask,
                )
                .unwrap();
                generation.bind_control_handle(&handle).unwrap();
                let mut prefix = vec![1, 4, 2, generation.next().unwrap().unwrap().token_id()];
                // Enter a draft/refinement transaction before requesting its handoff.
                if start == 2 {
                    assert!(matches!(
                        generation.advance_operation().unwrap().unwrap(),
                        TextAdvancement::Progress(_)
                    ));
                }
                handle.request_decoding_mode(requested).unwrap();
                loop {
                    match generation.advance_operation().unwrap().unwrap() {
                        TextAdvancement::Token(token) => prefix.push(token.token_id()),
                        TextAdvancement::Progress(_) => {}
                        TextAdvancement::ModeChanged(ack) => {
                            assert_eq!(ack.active, requested);
                            assert_eq!(ack.committed_position as usize, prefix.len() - 3);
                            break;
                        }
                    }
                }
                assert_ne!(generation.next().unwrap().unwrap().token_id(), 0);
                drop(generation);
            }
        }
    }
}

#[test]
fn loaded_nemotron_shared_drafts_preserve_causal_frontier_across_residency_policies() {
    let root = tiny_heterogeneous_artifact(config());
    let family =
        eredu_architectures::nemotron_labs_diffusion::FamilyConfig::from_hf_value(&config())
            .unwrap();
    let options = family
        .self_speculative_draft_options(
            eredu_core::decoding::SpeculationOptionsRequest {
                max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                ..Default::default()
            }
            .try_into()
            .unwrap(),
            0.0,
        )
        .unwrap();
    let mut ordinary = None;
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let request = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::with_layers(policy.layers()),
            ),
        );
        let model = eredu_core::load_model(&backend, root.path(), request)
            .unwrap_or_else(|error| panic!("{policy:?}: {error}"));
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        assert!(matches!(
            runtime.session().speculative_capability(),
            eredu_core::SpeculativeCapability::Ready {
                draft_source: eredu_core::SpeculativeDraftSource::SharedBackbone
            }
        ));
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(4),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let tokens = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        match &ordinary {
            Some(expected) => assert_eq!(&tokens, expected),
            None => ordinary = Some(tokens),
        }
        runtime.reset().unwrap();
        let prompt = <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
            runtime.backend(),
            vec![1, 4, 2],
        )
        .unwrap();
        runtime
            .session_mut()
            .with_model_operation(|model| {
                let context =
                    crate::composition::mlx::speculative::SpeculativeExecutionStreams::single(
                        &stream,
                    );
                let mut executor = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                    model.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                );
                let mut cache = executor.new_cache(context)?;
                let (_, seed, _) = executor
                    .prefill(
                        SharedBackboneInput::new(
                            prompt,
                            options,
                            crate::backend::random::RandomState::with_seed(17)?,
                            crate::backend::random::RandomState::with_seed(29)?,
                        ),
                        &mut cache,
                        context,
                    )?
                    .into_parts();
                let checkpoint = executor.checkpoint(&cache)?;
                let canonical = cache.retained_arrays();
                let prefix = checkpoint.retained_arrays();
                assert!(!canonical.is_empty());
                assert_eq!(canonical.len(), prefix.len());
                let survey = safemlx::inspect_storage(&canonical.iter().chain(&prefix).copied().collect::<Vec<_>>())?;
                let width = canonical.len();
                assert!(survey.values[..width].iter().all(Option::is_some));
                assert_eq!(&survey.values[..width], &survey.values[width..], "settled shared-prefix checkpoints introduce no KV backing replica");

                assert_eq!(frontier(&executor, &cache, &seed), 3);
                for capacity in [1, 2] {
                    let mut draft = executor.begin_proposal(&seed, 5, capacity, context)?;
                    assert_eq!(frontier(&executor, &cache, &seed), 3);
                    let mut proposed = vec![5];
                    for _ in 0..capacity {
                        let SpeculativeProposalStep::TargetMatchingToken(token) = executor
                            .proposal_step(&mut draft, *proposed.last().unwrap(), context)?
                        else {
                            panic!("masked draft invented a causal q")
                        };
                        proposed.push(token);
                    }
                    let verification =
                        executor.submit_verification(&proposed, &mut cache, context)?;
                    verification.completion.wait()?;
                    assert_eq!(frontier(&executor, &cache, &seed), 4 + capacity as u64);
                    let committed = executor.commit_verification(
                        verification.output,
                        draft,
                        &mut cache,
                        &checkpoint,
                        1,
                        context,
                    )?;
                    assert_eq!(
                        committed.into_parts().1,
                        0,
                        "full-attention K/V truncates without replay"
                    );
                    assert_eq!(frontier(&executor, &cache, &seed), 4);
                    executor.restore_checkpoint(&mut cache, &checkpoint, context)?;
                    assert_eq!(frontier(&executor, &cache, &seed), 3);
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn loaded_pixtral_context_survives_shared_backbone_drafting_in_bounded_execution() {
    let fixture: serde_json::Value = serde_json::from_str(
        eredu_evaluation::nemotron_labs_diffusion::PIXTRAL_REFERENCE_JSON,
    )
    .unwrap();
    let mut config = fixture["config"].clone();
    config["block_size"] = 3.into();
    let root = tiny_heterogeneous_artifact(config.clone());
    let family =
        eredu_architectures::nemotron_labs_diffusion::FamilyConfig::from_hf_value(&config).unwrap();
    let options = family
        .self_speculative_draft_options(
            eredu_core::decoding::SpeculationOptionsRequest {
                max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
                ..Default::default()
            }
            .try_into()
            .unwrap(),
            0.0,
        )
        .unwrap();
    let mut expected = None;
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let request = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::with_layers(policy.layers()),
            ),
        );
        let model = eredu_core::load_model(&backend, root.path(), request)
            .unwrap_or_else(|error| panic!("{policy:?}: {error}"));
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let patches = (0..48)
            .map(|i| (i as f32 - 23.0) / 37.0)
            .collect::<Vec<_>>();
        let parts = [
            input::token_ids_part(&Array::from_slice(&[1u32, 4], &[1, 2])).unwrap(),
            input::input_part(
                eredu_core::InputModality::Image,
                input::InputPayload::Tensor(Array::from_slice(&patches, &[4, 12])),
                [(
                    eredu_core::InputMetadataKey::PatchGrid,
                    Array::from_slice(&[1i32, 2, 2], &[1, 3]),
                )],
                [eredu_core::InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
            input::token_ids_part(&Array::from_slice(&[2u32], &[1, 1])).unwrap(),
        ];
        let prompt = MlxModelInput::from(input::ModelInput::new(&parts));
        runtime
            .session_mut()
            .with_model_operation(|model| {
                let context =
                    crate::composition::mlx::speculative::SpeculativeExecutionStreams::single(
                        &stream,
                    );
                let mut executor = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                    model.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                );
                let mut cache = executor.new_cache(context)?;
                let (scores, seed, positions) = executor
                    .prefill(
                        SharedBackboneInput::new(
                            prompt,
                            options,
                            crate::backend::random::RandomState::with_seed(17)?,
                            crate::backend::random::RandomState::with_seed(29)?,
                        ),
                        &mut cache,
                        context,
                    )?
                    .into_parts();
                assert_eq!(positions, 6);
                assert_eq!(
                    frontier(&executor, &cache, &seed),
                    6,
                    "image markers are part of the canonical causal prefix"
                );
                let mut draft = executor.begin_proposal(&seed, 5, 2, context)?;
                let mut proposed = vec![5];
                for _ in 0..2 {
                    let SpeculativeProposalStep::TargetMatchingToken(token) =
                        executor.proposal_step(&mut draft, *proposed.last().unwrap(), context)?
                    else {
                        unreachable!()
                    };
                    proposed.push(token);
                }
                assert_eq!(frontier(&executor, &cache, &seed), 6);
                let actual = (
                    scores
                        .into_evaluated()?
                        .try_to_vec::<f32>()
                        .map_err(|e| Error::ArchitectureModel(e.to_string()))?,
                    proposed,
                );
                match &expected {
                    Some(expected) => assert_eq!(&actual, expected),
                    None => expected = Some(actual),
                }
                Ok(())
            })
            .unwrap();
    }
}

impl eredu_runtime::execution_control::SnapshotTokenController for ExcludeMask {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(Self)
    }
}

#[test]
fn shared_target_matching_preserves_stochastic_sampler_and_evolving_penalties() {
    use eredu_core::decoding::{DecodingMode, SpeculationOptionsRequest};
    let root = tiny_heterogeneous_artifact(config());
    let (stream, weights) = execution_streams();
    let backend = crate::native::backend(&stream, &weights);
    let model = eredu_core::load_model(&backend, root.path(), Default::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(13),
            temperature: Some(0.8),
            repetition_penalty: Some(1.2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.2),
            ..Default::default()
        },
    )
    .unwrap();
    let mode = DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        }
        .try_into()
        .unwrap(),
    );
    for seed in [7, 13, 29] {
        runtime.reset().unwrap();
        let ar = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling).with_seed(seed),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        runtime.reset().unwrap();
        let shared = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling)
                .with_seed(seed)
                .with_decoding_mode(Some(mode)),
            ExcludeMask,
        )
        .unwrap()
        .map(|token| token.unwrap().token_id())
        .collect::<Vec<_>>();
        assert_eq!(
            shared, ar,
            "proposal draws must not consume the processed causal target stream"
        );
    }
}

#[test]
fn shared_transactions_capture_physical_verification_and_replay_isolated_snapshot_branches() {
    use eredu_core::{
        TextAdvancement, TextGenerationBackend, TextGenerationDriver,
        capture::*,
        decoding::{DecodingMode, SpeculationOptionsRequest},
        execution_control::SnapshotLimits,
    };
    use eredu_runtime::execution_control::{
        ManagedTextContinuation, SnapshotBudget, TextBranchRequest, TextContinuationSnapshot,
    };
    let root = tiny_heterogeneous_artifact(config());
    let (stream, weights) = execution_streams();
    let backend = crate::native::backend(&stream, &weights);
    let model = eredu_core::load_model(&backend, root.path(), Default::default()).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let mode = DecodingMode::SelfSpeculation(
        SpeculationOptionsRequest {
            max_draft_tokens: std::num::NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        }
        .try_into()
        .unwrap(),
    );
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(9),
            temperature: Some(0.8),
            ..Default::default()
        },
    )
    .unwrap();
    let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
    let point = discovery
        .catalog
        .points
        .iter()
        .find(|point| point.path == "model.layers.0.feed_forward.residual")
        .unwrap();
    let allowance = CaptureUsage {
        captures: 1000,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let limits = CaptureLimits {
        per_step: allowance,
        cumulative: allowance,
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Fail,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        provenance: CaptureProvenance::ContentExact,
        selections: vec![CaptureSelection {
            id: "verified-residual".into(),
            path: point.path.clone(),
            schedule: Default::default(),
            transform: CaptureTransform::FullTensor,
            slices: vec![],
        }],
        limits: limits.clone(),
    }
    .admit_invocations(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        MlxBackend::text_capture_invocation_bounds(
            &runtime,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 9,
            },
        )
        .unwrap()
        .unwrap(),
    )
    .unwrap();
    let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 4, 2]).unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut continuation = driver
        .start(
            prompt,
            eredu_core::TextGenerationConfig::new(sampling)
                .with_seed(13)
                .with_decoding_mode(Some(mode)),
            ExcludeMask,
        )
        .unwrap();
    driver.enable_capture(&mut continuation, capture).unwrap();
    let mut state = ManagedTextContinuation::root(continuation);
    assert!(matches!(
        state.advance_operation(&mut driver).unwrap().unwrap(),
        TextAdvancement::Token(_)
    ));
    state.take_completed_step(&mut driver).unwrap();
    assert!(matches!(
        state.advance_operation(&mut driver).unwrap().unwrap(),
        TextAdvancement::Progress(_)
    ));
    let captured = state
        .take_completed_step(&mut driver)
        .unwrap()
        .expect("verification capture");
    assert_eq!(captured.invocation.unwrap().sequence, 3);
    assert_eq!(captured.records.len(), 1);
    assert_eq!(captured.records[0].outcome, CaptureOutcome::Captured);
    assert!(matches!(
        state.advance_operation(&mut driver).unwrap().unwrap(),
        TextAdvancement::Token(_)
    ));
    state.take_completed_step(&mut driver).unwrap();
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 512 << 20,
    });
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(&mut driver).unwrap(),
        &budget,
        Some(4096),
    )
    .unwrap();
    let growth = saved
        .native_continuation_growth(driver.runtime(), 9)
        .unwrap();
    let mut child = saved
        .fork(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "shared-fork",
                max_predictions: 9,
                capture_limits: Some(limits),
                intervention: None,
                host_bytes: Some(4096),
                continuation_growth_bytes: Some(growth + (1 << 20)),
            },
        )
        .unwrap();
    fn collect<'a>(
        driver: &mut TextGenerationDriver<'_, MlxBackend<'a>>,
        state: &mut ManagedTextContinuation<MlxBackend<'a>, ExcludeMask>,
    ) -> Vec<u32> {
        let mut tokens = Vec::new();
        while let Some(step) = state.advance_operation(driver).unwrap() {
            state.take_completed_step(driver).unwrap();
            if let TextAdvancement::Token(token) = step {
                tokens.push(token.token_id());
            }
        }
        tokens
    }
    let expected = collect(&mut driver, &mut state);
    let copy_before = budget.usage().cumulative_copy_bytes;
    saved
        .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
        .unwrap();
    assert_eq!(collect(&mut driver, &mut state), expected);
    assert!(budget.usage().cumulative_copy_bytes > copy_before);
    child.exchange(&mut driver, &mut state).unwrap();
    assert_eq!(collect(&mut driver, &mut state), expected);
    child.exchange(&mut driver, &mut state).unwrap();
}
