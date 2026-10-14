use super::*;
use crate::composition::mlx::speculative::{MlxSpeculativeSampling, SpeculativeExecutionStreams};
use eredu_core::decoding::{DecodingMode, SpeculationOptions};
use eredu_core::{
    GenerationRequestConfig, GenerationRequestTable, SpeculativeCallbackPublisher,
    SpeculativeOutputRuntime, SpeculativeRandomness, SpeculativeSemanticConstraint,
};
use eredu_runtime::ConstrainedSampler;

/// Initial diffusion emits no causal seed token. Complete block suffixes settle
/// once before all six directed handoffs, and reusable snapshots retain both
/// the suffix and independently spent numerical streams.
#[test]
fn shared_request_table_refines_initial_blocks_and_all_six_handoffs() {
    let root = tiny_heterogeneous_artifact(config());
    for residency in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let loaded = eredu_core::load_model(
            &backend,
            root.path(),
            crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                    eredu_runtime::WeightResidency::with_layers(residency.layers()),
                ),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, loaded).unwrap();
        let sampling = eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(3),
                temperature: Some(0.0),
                repetition_penalty: Some(1.0),
                ..Default::default()
            },
        )
        .unwrap();
        let diffusion = DecodingMode::Diffusion { policy: None };
        let expected_block = ControlledTextGeneration::new(
            &mut runtime,
            vec![1, 4, 2],
            eredu_core::TextGenerationConfig::new(sampling).with_decoding_mode(Some(diffusion)),
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
                    .with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap());
                let draft = model
                    .erased()
                    .capability_estimate()
                    .shared_backbone_draft()
                    .unwrap()
                    .prepare(options, 0.0)
                    .unwrap();
                let mut typed = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                    model.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                );
                let mut executor = eredu_core::speculative::DynGenerationExecutor::<
                    crate::composition::mlx::prepared_speculative::MlxSharedGenerationExecutorTypes,
                >::new(&mut typed);
                let mut cache = executor.new_cache(context)?;
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(
                        eredu_runtime::DefaultSampler,
                        ExcludeMask,
                    )),
                    eredu_core::GenerationSequence::new(32, []),
                    SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|_| Ok(())),
                    Default::default(),
                );
                let mut table =
                    GenerationRequestTable::new(Default::default(), context.topology()).unwrap();
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
                            max_tokens: 32,
                            decoding_mode: diffusion,
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
                assert!(table.request(id).unwrap().sequence().tokens().is_empty());
                table.step(&mut executor, false, context).unwrap();
                assert!(table.request(id).unwrap().sequence().tokens().is_empty());
                assert_eq!(table.request(id).unwrap().stats().refinement_steps(), 1);
                assert!(
                    table
                        .request(id)
                        .unwrap()
                        .stats()
                        .submission_to_first_token()
                        .is_none()
                );
                assert!(table.request(id).unwrap().has_tentative_refinement());
                let budget = eredu_runtime::execution_control::SnapshotBudget::new(
                    eredu_core::execution_control::SnapshotLimits {
                        max_snapshots: 2,
                        max_branches: 0,
                        retained_bytes: 128 << 20,
                        cumulative_copy_bytes: 512 << 20,
                    },
                );
                let partial_estimate = table
                    .request(id)
                    .unwrap()
                    .control_snapshot_estimate(&executor)
                    .unwrap();
                let _partial_reservation = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Snapshot,
                        Some(partial_estimate),
                    )
                    .unwrap();
                let partial = table
                    .request(id)
                    .unwrap()
                    .control_snapshot(&executor, context)
                    .unwrap();
                table
                    .request(id)
                    .unwrap()
                    .request_decoding_mode(DecodingMode::Autoregressive)
                    .unwrap();
                assert_eq!(
                    table.request(id).unwrap().request_decoding_mode(diffusion),
                    Err(eredu_core::decoding::DecodingModeError::ModeChangePending)
                );
                while table
                    .request(id)
                    .unwrap()
                    .decoding_state()
                    .pending
                    .is_some()
                {
                    assert_eq!(
                        table.request(id).unwrap().decoding_state().active,
                        diffusion
                    );
                    table.step(&mut executor, false, context).unwrap();
                }
                assert_eq!(
                    table
                        .request(id)
                        .unwrap()
                        .decoding_state()
                        .acknowledgement
                        .unwrap()
                        .committed_position,
                    3
                );
                let _partial_restore = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Restore,
                        Some(partial_estimate),
                    )
                    .unwrap();
                table
                    .request_mut(id)
                    .unwrap()
                    .restore_control_snapshot(&mut executor, &partial, context)
                    .unwrap();
                assert!(table.request(id).unwrap().has_tentative_refinement());
                while table.request(id).unwrap().has_tentative_refinement() {
                    table.step(&mut executor, false, context).unwrap();
                }
                assert_eq!(
                    table.request(id).unwrap().sequence().tokens(),
                    expected_block
                );
                assert_eq!(
                    table.request(id).unwrap().pending_committed_tokens(),
                    expected_block
                );
                assert_eq!(
                    table
                        .request(id)
                        .unwrap()
                        .continuation_memory_observation(&executor, 0)
                        .unwrap()
                        .unwrap()
                        .target
                        .current_positions,
                    3
                );
                let settled_estimate = table
                    .request(id)
                    .unwrap()
                    .control_snapshot_estimate(&executor)
                    .unwrap();
                let _settled_reservation = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Snapshot,
                        Some(settled_estimate),
                    )
                    .unwrap();
                let saved = table
                    .request(id)
                    .unwrap()
                    .control_snapshot(&executor, context)
                    .unwrap();
                let modes = [
                    DecodingMode::Autoregressive,
                    DecodingMode::SelfSpeculation(options),
                    DecodingMode::Autoregressive,
                    diffusion,
                    DecodingMode::SelfSpeculation(options),
                    diffusion,
                ];
                let mut acknowledgements = Vec::new();
                for mode in modes {
                    let position = table.request(id).unwrap().sequence().tokens().len();
                    table
                        .request(id)
                        .unwrap()
                        .request_decoding_mode(mode)
                        .unwrap();
                    while table
                        .request(id)
                        .unwrap()
                        .decoding_state()
                        .pending
                        .is_some()
                    {
                        assert!(table.step(&mut executor, false, context).unwrap());
                    }
                    let request = table.request(id).unwrap();
                    assert_eq!(request.decoding_state().active, mode);
                    assert!(request.pending_committed_tokens().is_empty());
                    assert_eq!(
                        request
                            .decoding_state()
                            .acknowledgement
                            .unwrap()
                            .committed_position,
                        position as u64
                    );
                    assert_eq!(
                        request
                            .continuation_memory_observation(&executor, 0)
                            .unwrap()
                            .unwrap()
                            .target
                            .current_positions,
                        3 + position as u64
                    );
                    acknowledgements.push(position);
                    while table.request(id).unwrap().sequence().tokens().len() == position
                        || table.request(id).unwrap().has_tentative_refinement()
                    {
                        assert!(table.step(&mut executor, false, context).unwrap());
                    }
                }
                let expected = table.request(id).unwrap().sequence().tokens().to_vec();
                let _settled_restore = budget
                    .reserve(
                        eredu_core::execution_control::SnapshotResourceKind::Restore,
                        Some(settled_estimate),
                    )
                    .unwrap();
                table
                    .request_mut(id)
                    .unwrap()
                    .restore_control_snapshot(&mut executor, &saved, context)
                    .unwrap();
                for (mode, position) in modes.into_iter().zip(acknowledgements) {
                    assert_eq!(
                        table.request(id).unwrap().sequence().tokens().len(),
                        position
                    );
                    table
                        .request(id)
                        .unwrap()
                        .request_decoding_mode(mode)
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
                    while table.request(id).unwrap().sequence().tokens().len() == position
                        || table.request(id).unwrap().has_tentative_refinement()
                    {
                        table.step(&mut executor, false, context).unwrap();
                    }
                }
                assert_eq!(table.request(id).unwrap().sequence().tokens(), expected);
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn controlled_shared_diffusion_forks_partial_canvases_and_refines_forced_suffixes() {
    use eredu_core::{PreparedSpeculativeLane, SpeculativeGenerationVisitor};
    use eredu_runtime::memory_forecast::GenerationForecastBackend;
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let root = tiny_heterogeneous_artifact(config());
    for residency in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let loaded = eredu_core::load_model(
            &backend,
            root.path(),
            crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                    eredu_runtime::WeightResidency::with_layers(residency.layers()),
                ),
            ),
        )
        .unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, loaded).unwrap();
        let profiles = Some((
            MlxBackend::loaded_memory_profile(&runtime).unwrap(),
            MlxBackend::shared_backbone_memory_profile(&runtime)
                .unwrap()
                .unwrap(),
        ));
        let prompt = <MlxBackend<'_> as eredu_core::TextGenerationBackend>::prepare_text_prompt(
            runtime.backend(),
            vec![1, 4, 2],
        )
        .unwrap();
        let discovery =
            <MlxBackend<'_> as eredu_core::TextGenerationBackend>::capture_discovery(&runtime)
                .unwrap();
        let capture = eredu_core::capture::CapturePlan {
            schema_version: eredu_core::capture::CAPTURE_SCHEMA_VERSION,
            provenance: eredu_core::capture::CaptureProvenance::ContentExact,
            selections: vec![eredu_core::capture::CaptureSelection {
                id: "position-scores".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: Default::default(),
                transform: eredu_core::capture::CaptureTransform::FullTensor,
                slices: vec![],
            }],
            limits: eredu_core::capture::CaptureLimits {
                per_step: eredu_core::capture::CaptureUsage {
                    captures: 10,
                    retained_bytes: 1 << 20,
                    host_bytes: 1 << 20,
                    encoded_bytes: 1 << 20,
                },
                cumulative: eredu_core::capture::CaptureUsage {
                    captures: 100,
                    retained_bytes: 16 << 20,
                    host_bytes: 16 << 20,
                    encoded_bytes: 16 << 20,
                },
                physical_native_bytes: None,
                on_limit: eredu_core::capture::CaptureLimitPolicy::Fail,
            },
        }
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            eredu_core::capture::CaptureRequestShape {
                batch: 1,
                prompt_tokens: 1,
                max_predictions: 8,
            },
        )
        .unwrap();
        let internal_discovery = <MlxBackend<'_> as eredu_core::SpeculativeGenerationBackend>::speculative_activation_discovery(&runtime).unwrap();
        let activations = eredu_core::speculative::SpeculativeActivationPlan {
            schema_version: eredu_core::speculative::SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            bounds: eredu_core::capture::CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: None,
                max_predictions: 8,
            },
            captures: eredu_core::capture::CapturePlan {
                schema_version: eredu_core::capture::CAPTURE_SCHEMA_VERSION,
                provenance: eredu_core::capture::CaptureProvenance::ContentExact,
                selections: vec![eredu_core::capture::CaptureSelection {
                    id: "shared-layer".into(),
                    path: "model.layers.0.feed_forward.residual".into(),
                    schedule: Default::default(),
                    transform: eredu_core::capture::CaptureTransform::FullTensor,
                    slices: vec![],
                }],
                limits: eredu_core::capture::CaptureLimits {
                    per_step: eredu_core::capture::CaptureUsage {
                        captures: 4,
                        retained_bytes: 1 << 20,
                        host_bytes: 1 << 20,
                        encoded_bytes: 1 << 20,
                    },
                    cumulative: eredu_core::capture::CaptureUsage {
                        captures: 100,
                        retained_bytes: 16 << 20,
                        host_bytes: 16 << 20,
                        encoded_bytes: 16 << 20,
                    },
                    physical_native_bytes: None,
                    on_limit: eredu_core::capture::CaptureLimitPolicy::Fail,
                },
            },
            interventions: eredu_core::intervention::InterventionPlan::none(),
        }
        .admit(&internal_discovery)
        .unwrap();
        let mut failure = None;
        let options = ControlledSpeculativeOptions {
            capture: Some(capture),
            activations: Some(activations),
            snapshots: Some(eredu_core::execution_control::SnapshotLimits {
                max_snapshots: 2,
                max_branches: 1,
                retained_bytes: 128 << 20,
                cumulative_copy_bytes: 512 << 20,
            }),
            ..Default::default()
        };
        runtime
            .session_mut()
            .with_model_operation(|model| {
                let context = SpeculativeExecutionStreams::single(&stream);
                let speculation = SpeculationOptions::default()
                    .with_proposal_capacity(std::num::NonZeroUsize::new(2).unwrap());
                let draft = model
                    .erased()
                    .capability_estimate()
                    .shared_backbone_draft()
                    .unwrap()
                    .prepare(speculation, 0.0)
                    .unwrap();
                let mut typed = SharedBackboneExecutor::<MlxAutoregressiveMechanisms>::new(
                    model.erased_mut(),
                    std::num::NonZeroUsize::new(2).unwrap(),
                );
                let mut executor = eredu_core::speculative::DynGenerationExecutor::<
                    crate::composition::mlx::prepared_speculative::MlxSharedGenerationExecutorTypes,
                >::new(&mut typed);
                let cache = executor.new_cache(context)?;
                let output = SpeculativeOutputRuntime::new(
                    MlxSpeculativeSampling::new(ConstrainedSampler::new(
                        eredu_runtime::DefaultSampler,
                        ExcludeMask,
                    )),
                    eredu_core::GenerationSequence::new(8, []),
                    SpeculativeSemanticConstraint::plain(),
                    SpeculativeCallbackPublisher::tokens(|_| Ok(())),
                    Default::default(),
                );
                let lane = PreparedSpeculativeLane::new(
                    eredu_core::speculative::GenerationCache::Owned(cache),
                    SharedBackboneInput::new(
                        prompt,
                        draft,
                        crate::backend::random::RandomState::with_seed(17)?,
                        crate::backend::random::RandomState::with_seed(29)?,
                    ),
                    GenerationRequestConfig {
                        max_tokens: 8,
                        decoding_mode: DecodingMode::Diffusion { policy: None },
                        ..Default::default()
                    },
                    output,
                    SpeculativeRandomness::new(None, None),
                );
                let visitor = DriveControlledSpeculation::new(
                    Default::default(),
                    options,
                    |session: &mut dyn ControlledSpeculativeSession| {
                        let prefill = session.step()?.unwrap();
                        assert!(prefill.committed_token_ids.is_empty());
                        assert_eq!(prefill.activations.len(), 1);
                        assert_eq!(
                            prefill.activations[0].phase,
                            eredu_core::speculative::SpeculativeActivationPhase::TargetPrefill
                        );
                        let numerical = session.step()?.unwrap();
                        assert!(numerical.committed_token_ids.is_empty());
                        assert!(numerical.captures.is_empty());
                        assert!(numerical.timing.time_to_first_token().is_none());
                        assert_eq!(numerical.activations.len(), 1);
                        assert_eq!(numerical.activations[0].phase, eredu_core::speculative::SpeculativeActivationPhase::Refinement { iteration: 0 });
                        assert_eq!(numerical.refinement.as_ref().unwrap().finalized_positions, 0);
                        let numerical_saved = session.snapshot()?;
                        let next_iteration = session.step()?.unwrap();
                        assert!(next_iteration.committed_token_ids.is_empty());
                        assert_eq!(next_iteration.activations[0].phase, eredu_core::speculative::SpeculativeActivationPhase::Refinement { iteration: 1 });
                        let paid = session.snapshot_usage().cumulative_copy_bytes;
                        session.restore(&numerical_saved)?;
                        let repeated = session.step()?.unwrap();
                        assert_eq!(repeated.activations[0].phase, next_iteration.activations[0].phase);
                        assert!(repeated.activations[0].invocation > next_iteration.activations[0].invocation);
                        assert!(session.snapshot_usage().cumulative_copy_bytes > paid);
                        session.release_snapshot(&numerical_saved)?;
                        let mut last_invocation = repeated.activations[0].invocation;
                        let first = loop {
                            let step = session.step()?.unwrap();
                            for record in &step.activations {
                                assert!(matches!(record.phase, eredu_core::speculative::SpeculativeActivationPhase::Refinement { .. }));
                                assert_eq!(record.origin.committed_tokens, 0);
                                assert_eq!(record.captures.invocation.unwrap().sequence, 3);
                                assert!(record.invocation > last_invocation);
                                last_invocation = record.invocation;
                            }
                            if !step.committed_token_ids.is_empty() { break step; }
                        };
                        assert_eq!(first.committed_token_ids.len(), 1);
                        assert_eq!(first.captures.len(), 1);
                        assert_eq!(first.captures[0].score_kind, eredu_core::refinement::PredictionScoreKind::RefinementPosition);
                        assert!(first.timing.time_to_first_token().is_some());
                        let canvas = first.refinement.unwrap();
                        assert_eq!(canvas.finalized_positions, 1);
                        assert_eq!(canvas.canvas_positions, 3);
                        let selected = canvas.token_ids.unwrap()[0];
                        let forced = (selected % 10) + 1;
                        assert_ne!(forced, selected);
                        let saved = session.snapshot()?;
                        let child = session.fork(&saved)?;
                        session.force_next_token(forced)?;
                        let replaced = session.step()?.unwrap();
                        assert_eq!(replaced.committed_token_ids, [forced]);
                        assert!(replaced.refinement.unwrap().token_ids.is_none());
                        session.request_decoding_mode(DecodingMode::Autoregressive)?;
                        let forecast = session
                            .forecast_remaining_generation(0, &Default::default())
                            .unwrap();
                        assert_eq!(forecast.continuation.settlement_input_tokens, 3);
                        assert!(
                            forecast
                                .estimate
                                .domains
                                .iter()
                                .flat_map(|domain| &domain.phases)
                                .any(|phase| phase.phase
                                    == eredu_runtime::memory_estimation::MemoryPhase::Refinement)
                        );
                        while session.token_ids().len() < 3 {
                            let refined = session.step()?.unwrap();
                            for record in &refined.activations {
                                assert!(matches!(record.phase, eredu_core::speculative::SpeculativeActivationPhase::Refinement { .. }));
                                assert_eq!(record.origin.committed_tokens, 2);
                                assert!(record.invocation > last_invocation);
                                last_invocation = record.invocation;
                            }
                        }
                        let settled = session.step()?.unwrap();
                        assert_eq!(settled.activations.len(), 1);
                        assert_eq!(
                            settled.activations[0].phase,
                            eredu_core::speculative::SpeculativeActivationPhase::Verification
                        );
                        assert!(settled.activations[0].invocation > last_invocation);
                        assert_eq!(
                            session
                                .decoding_state()
                                .unwrap()
                                .acknowledgement
                                .unwrap()
                                .committed_position,
                            3
                        );
                        let causal = session.step()?.unwrap();
                        assert_eq!(causal.captures.len(), 1);
                        assert_eq!(
                            causal.captures[0].score_kind,
                            eredu_core::refinement::PredictionScoreKind::CausalConditional
                        );
                        let prefix = session.token_ids().to_vec();
                        let spent = session.snapshot_usage().cumulative_copy_bytes;
                        session.exchange(&child)?;
                        assert_eq!(session.token_ids(), &prefix[..1]);
                        assert_eq!(
                            session.decoding_state().unwrap().active,
                            DecodingMode::Diffusion { policy: None }
                        );
                        assert!(session.snapshot_usage().cumulative_copy_bytes > spent);
                        let resumed = session.step()?.unwrap();
                        assert_eq!(resumed.committed_token_ids, [selected]);
                        assert!(resumed.activations.is_empty());
                        assert!(resumed.sequence > replaced.sequence);
                        session
                            .request_decoding_mode(DecodingMode::SelfSpeculation(speculation))?;
                        while session.decoding_state().unwrap().pending.is_some() {
                            session.step()?;
                        }
                        assert_eq!(
                            session
                                .decoding_state()
                                .unwrap()
                                .acknowledgement
                                .unwrap()
                                .committed_position,
                            3
                        );
                        session.step()?; // Sample the settled target row before drafting its successor.
                        let drafted = session.step()?.unwrap();
                        assert!(drafted.activations.iter().any(|record| matches!(
                            record.phase,
                            eredu_core::speculative::SpeculativeActivationPhase::MaskedDraft { .. }
                        )
                            && record.origin.committed_tokens == 4
                            && record.invocation > last_invocation
                            && record.captures.invocation.unwrap().sequence == 3));
                        let verified = session.step()?.unwrap();
                        assert!(verified.activations.iter().any(|record| record.phase
                            == eredu_core::speculative::SpeculativeActivationPhase::Verification));
                        session.release_branch(&child)?;
                        session.release_snapshot(&saved)?;
                        session.cancel()?;
                        Ok(())
                    },
                    &mut failure,
                )
                .with_vocabulary(11)
                .with_forecast_profiles(profiles);
                visitor
                    .run(
                        &mut executor,
                        vec![lane],
                        context.topology(),
                        false,
                        false,
                        context,
                    )
                    .map_err(|error| {
                        Error::Exception(safemlx::error::Exception::from_source(error))
                    })?;
                Ok(())
            })
            .unwrap();
        assert!(failure.is_none(), "{failure:?}");
    }
}
