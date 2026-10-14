//! The causal draft source retains both caches through ordinary-mode handoffs.
use super::*;

#[test]
fn independent_causal_source_handoffs_preserve_both_frontiers_and_target_output() {
    use crate::composition::mlx::speculative::{
        MlxSpeculativeSampling, SpeculativeExecutionStreams,
    };
    use eredu_core::{
        GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
        SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
        decoding::{DecodingMode, DecodingModeError, SpeculationOptions},
        generation::SpeculativeRequestStatus,
    };
    use eredu_runtime::{ConstrainedSampler, speculative::autoregressive::AutoregressiveExecutor};
    let target_root = tiny_heterogeneous_artifact(config());
    let mut draft_config = config();
    draft_config["hidden_size"] = 5.into();
    let draft_root = tiny_heterogeneous_artifact(draft_config);
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let load = || {
            crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                    eredu_runtime::WeightResidency::with_layers(policy.layers()),
                ),
            )
        };
        let target = eredu_core::load_model(&backend, target_root.path(), load()).unwrap();
        let draft = eredu_core::load_model(&backend, draft_root.path(), load()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, target).unwrap();
        let mut draft = draft.into_inner().into_executable();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                ..Default::default()
            },
        )
        .unwrap();
        let config = eredu_core::TextGenerationConfig::new(sampling);
        let sampler =
            crate::composition::mlx::session::MlxTextSampler::from_config(&config).unwrap();
        let baseline =
            ControlledTextGeneration::new(&mut runtime, vec![1, 4, 2], config, ExcludeMask)
                .unwrap()
                .map(|token| token.unwrap().token_id())
                .collect::<Vec<_>>();
        runtime.reset().unwrap();
        let prompt = <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
            runtime.backend(),
            vec![1, 4, 2],
        )
        .unwrap();
        let transition_prompt =
            <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
                runtime.backend(),
                vec![1, 4, 2],
            )
            .unwrap();
        runtime
            .session_mut()
            .with_model_operation(|target| {
                let context = SpeculativeExecutionStreams::single(&stream);
                let options = SpeculationOptions::default()
                    .with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap());
                let mut executor = AutoregressiveExecutor::<MlxAutoregressiveMechanisms, eredu_runtime::speculative::autoregressive::SharedTargetRefinement>::new(
                    target.erased_mut(),
                    draft.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                    eredu_runtime::speculative::autoregressive::SharedTargetRefinement,
                );
                let mut cache = executor.new_cache(context)?;
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(sampler.clone(), ExcludeMask)),
                    eredu_core::GenerationSequence::new(8, []),
                    SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|_| Ok(())),
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
                        eredu_runtime::speculative::autoregressive::AutoregressiveInput::new(prompt,
                            eredu_runtime::speculative::autoregressive::SharedTargetRefinementState::new(crate::backend::random::RandomState::with_seed(29)?)),
                        GenerationRequestConfig {
                            max_tokens: 8,
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
                let request = table.request(id).unwrap();
                assert_eq!(
                    request.decoding_state().active,
                    DecodingMode::Autoregressive
                );
                assert_eq!(
                    request.request_decoding_mode(DecodingMode::SelfSpeculation(
                        options.with_proposal_capacity(std::num::NonZeroUsize::new(3).unwrap())
                    )),
                    Err(DecodingModeError::Unsupported("proposal capacity exceeds retained draft geometry"))
                );
                assert_eq!(request.decoding_state().pending, None);
                request
                    .request_decoding_mode(DecodingMode::SelfSpeculation(options))
                    .unwrap();
                table.step(&mut executor, false, context).unwrap();
                let request = table.request(id).unwrap();
                let native = request
                    .continuation_memory_observation(&executor, 0)
                    .unwrap()
                    .unwrap();
                assert_eq!(native.target.current_positions, 4);
                assert_eq!(native.draft.unwrap().current_positions, 4);
                assert_eq!(
                    request
                        .decoding_state()
                        .acknowledgement
                        .unwrap()
                        .committed_position,
                    1
                );
                table.step(&mut executor, false, context).unwrap();
                // Retained target scores commit the second token without another input.
                let native = table
                    .request(id)
                    .unwrap()
                    .continuation_memory_observation(&executor, 0)
                    .unwrap()
                    .unwrap();
                assert_eq!(native.target.current_positions, 4);
                assert_eq!(native.draft.unwrap().current_positions, 4);
                table.step(&mut executor, false, context).unwrap();
                assert!(table.request(id).unwrap().stats().draft_tokens() > 0);
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::Autoregressive)
                    .unwrap();
                while table
                    .request(id)
                    .unwrap()
                    .decoding_state()
                    .pending
                    .is_some()
                {
                    table.step(&mut executor, false, context).unwrap();
                }
                let request = table.request(id).unwrap();
                assert_eq!(request.status(), SpeculativeRequestStatus::ReadyToDraft);
                let committed = request.sequence().tokens().len() as u64;
                assert_eq!(
                    request
                        .decoding_state()
                        .acknowledgement
                        .unwrap()
                        .committed_position,
                    committed
                );
                let native = request
                    .continuation_memory_observation(&executor, 0)
                    .unwrap()
                    .unwrap();
                assert_eq!(native.target.current_positions, committed + 3);
                assert_eq!(native.draft.unwrap().current_positions, committed + 3);
                table.run(&mut executor, false, context).unwrap();
                assert_eq!(table.request(id).unwrap().sequence().tokens(), baseline);
                drop(table);
                let mut cache = executor.new_cache(context)?;
                let prompt = transition_prompt;
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(sampler, ExcludeMask)),
                    eredu_core::GenerationSequence::new(32, []), SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|_| Ok(())), Default::default());
                let mut table = GenerationRequestTable::new(Default::default(), context.topology()).unwrap();
                let id = table.submit(&mut executor, &mut cache,
                    eredu_runtime::speculative::autoregressive::AutoregressiveInput::new(prompt,
                        eredu_runtime::speculative::autoregressive::SharedTargetRefinementState::new(crate::backend::random::RandomState::with_seed(29)?)),
                    GenerationRequestConfig { max_tokens: 32, ..Default::default() }, output,
                    SpeculativeRandomness::new(None,None), false, context).unwrap();
                assert_eq!(executor.verification_policy(), eredu_core::speculative::SpeculativeVerificationPolicy::LikelihoodRatio);
                let diffusion = DecodingMode::Diffusion { policy: None };
                let modes = [diffusion, DecodingMode::Autoregressive, DecodingMode::SelfSpeculation(options),
                    diffusion, DecodingMode::SelfSpeculation(options), DecodingMode::Autoregressive];
                for mode in modes {
                    let before = table.request(id).unwrap().sequence().tokens().len();
                    let drafted_before = table.request(id).unwrap().stats().draft_tokens();
                    table.request(id).unwrap().request_decoding_mode(mode).unwrap();
                    while table.request(id).unwrap().decoding_state().pending.is_some() { table.step(&mut executor, false, context).unwrap(); }
                    let request = table.request(id).unwrap();
                    let position = request.sequence().tokens().len();
                    assert_eq!(request.decoding_state().active, mode);
                    assert_eq!(request.decoding_state().acknowledgement.unwrap().committed_position, position as u64);
                    assert!(request.pending_committed_tokens().is_empty());
                    let observation = request.continuation_memory_observation(&executor, 0).unwrap().unwrap();
                    assert_eq!(observation.target.current_positions, 3 + position as u64);
                    assert_eq!(observation.draft.unwrap().current_positions, 3 + position as u64);
                    let mut checked_snapshot = false;
                    while table.request(id).unwrap().sequence().tokens().len() == before || table.request(id).unwrap().has_tentative_refinement() {
                        table.step(&mut executor, false, context).unwrap();
                        if mode == diffusion && !checked_snapshot && table.request(id).unwrap().sequence().tokens().len() == before {
                            let budget = eredu_runtime::execution_control::SnapshotBudget::new(eredu_core::execution_control::SnapshotLimits {
                                max_snapshots: 1, max_branches: 0, retained_bytes: 128 << 20, cumulative_copy_bytes: 512 << 20 });
                            let estimate = table.request(id).unwrap().control_snapshot_estimate(&executor).unwrap();
                            let _saved_budget = budget.reserve(eredu_core::execution_control::SnapshotResourceKind::Snapshot, Some(estimate)).unwrap();
                            let saved = table.request(id).unwrap().control_snapshot(&executor, context).unwrap();
                            table.step(&mut executor, false, context).unwrap();
                            let _restore_budget = budget.reserve(eredu_core::execution_control::SnapshotResourceKind::Restore, Some(estimate)).unwrap();
                            table.request_mut(id).unwrap().restore_control_snapshot(&mut executor, &saved, context).unwrap();
                            assert_eq!(table.request(id).unwrap().sequence().tokens().len(), before);
                            assert!(table.request(id).unwrap().has_tentative_refinement());
                            checked_snapshot = true;
                        }
                    }
                    if matches!(mode, DecodingMode::SelfSpeculation(_)) {
                        while table.request(id).unwrap().stats().draft_tokens() == drafted_before
                            || table.request(id).unwrap().block().is_some()
                            || table.request(id).unwrap().has_pending_verification() {
                            table.step(&mut executor, false, context).unwrap();
                        }
                    }
                }
                assert!(table.request(id).unwrap().stats().refinement_steps() > 0);
                assert!(table.request(id).unwrap().stats().draft_tokens() > 0);
                Ok(())
            })
            .unwrap();
    }
}
