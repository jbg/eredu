use super::*;
use eredu_core::generation::SpeculativeRequestStatus;
use eredu_core::residency::{ParameterConversionRetentionPolicy, ParameterConversionTrimError};

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_speculative_continuation_forecasts_cover_settled_state_without_advancement() {
    let fixture = super::startup_forecast::attention_prediction_fixture(true);
    let path = std::env::var_os("EREDU_SPECULATIVE_CONTINUATION_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture.0.clone());
    let repetitions = std::env::var("EREDU_SPECULATIVE_CONTINUATION_REPEAT")
        .map(|n| n.parse::<usize>().unwrap())
        .unwrap_or(1);
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    let previous = set_local_allocator_cache_limit(0).unwrap();
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_allocator_cache_limit(self.0).unwrap();
        }
    }
    let _restore = Restore(previous);
    for lookahead in [false, true] {
        let plan = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_parameter_conversion_retention(Some(
                ParameterConversionRetentionPolicy::Bounded { max_bytes: 1024 },
            ))
            .with_drafting(DraftingPlan::External {
                model: path.display().to_string(),
                placement: if lookahead {
                    DraftPlacementPlan::Device {
                        device: local_device_plan(device).unwrap(),
                    }
                } else {
                    DraftPlacementPlan::Target
                },
                max_draft_tokens: 4,
                lookahead,
                adaptive_lookahead: false,
            });
        let mut loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &plan).unwrap();
        let speculative_options = loaded.speculative_generation_options().unwrap().unwrap();
        let (model, drafting) = loaded.parts_mut();
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":
                "The quick brown fox jumped over the fence. ".repeat(repetitions)})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let prompt = model.count_prepared_chat(&chat).unwrap().model_positions;
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(32),
                temperature: Some(if lookahead { 0.8 } else { 0.0 }),
                ..Default::default()
            },
            seed: 42,
            ..Default::default()
        };
        let expected = model
            .generate_prepared_text_speculative(PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                drafting: drafting.as_speculative_draft().unwrap(),
                settings,
                options: speculative_options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        let options = GenerationForecastOptions {
            budget: MemoryBudget {
                application_limit_bytes: Some(256 << 30),
                ..Default::default()
            },
            ..Default::default()
        };
        let output = model.with_controlled_text_speculative(
            PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::rendered_prompt(&chat),
                drafting: drafting.as_speculative_draft().unwrap(),
                settings,
                options: speculative_options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            },
            ControlledSpeculativeOptions {
                snapshots: Some(SnapshotLimits {
                    max_snapshots: 1,
                    max_branches: 1,
                    retained_bytes: 2 << 30,
                    cumulative_copy_bytes: 8 << 30,
                }),
                ..Default::default()
            },
            |session| {
                assert!(session.forecast_remaining_generation(16, &options).is_err());
                assert!(matches!(session.trim_parameter_conversions(),
                    Err(ParameterConversionTrimError::NotQuiescent)));
                session.step()?.unwrap();
                assert_eq!(session.status(), SpeculativeRequestStatus::ReadyToDraft);
                let tokens = session.token_ids().to_vec();
                let sampling = session.sampling_state();
                let epoch = session.epoch();
                let usage = session.snapshot_usage();
                let before = eredu_backend_mlx::allocator_memory().unwrap().active_bytes();
                let remaining = 32 - tokens.len() as u64;
                let forecast = session.forecast_remaining_generation(remaining, &options).unwrap();
                let zero = session.forecast_remaining_generation(0, &options).unwrap();
                let repeated = session.forecast_remaining_generation(remaining, &options).unwrap();
                assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
                assert_eq!(forecast.continuation.target.current_positions, prompt);
                assert_eq!(forecast.continuation.draft.as_ref().unwrap().current_positions, prompt);
                assert!(forecast.continuation.target.current_state.lower_bytes > 0);
                assert!(forecast.continuation.draft.as_ref().unwrap().current_state.lower_bytes > 0);
                assert_eq!(zero.continuation.target.current_state, forecast.continuation.target.current_state);
                assert_eq!(forecast.continuation, repeated.continuation);
                for pool in &forecast.estimate.domains {
                    assert!(pool.phases.iter().all(|p| !matches!(p.phase,
                        MemoryPhase::Loading | MemoryPhase::Prefill)));
                }
                assert_eq!(session.token_ids(), tokens);
                assert_eq!(session.sampling_state(), sampling);
                assert_eq!(session.epoch(), epoch);
                assert_eq!(session.snapshot_usage(), usage);
                // Completed work may release buffers asynchronously on separate streams.
                assert!(eredu_backend_mlx::allocator_memory().unwrap().active_bytes() <= before);

                let saved = session.snapshot()?;
                let branch = session.fork(&saved)?;
                let retained = session.forecast_remaining_generation(remaining, &options).unwrap();
                assert!(retained.continuation.retained_snapshots.upper_bytes.unwrap() > 0);
                session.exchange(&branch)?;
                let exchanged = session.forecast_remaining_generation(remaining, &options).unwrap();
                assert_eq!(exchanged.continuation.target.current_positions, prompt);
                let saved_usage = session.snapshot_usage();
                let trimmed = session.trim_parameter_conversions().unwrap();
                let draft_trimmed = trimmed.external_drafter.as_ref().unwrap();
                assert_eq!(trimmed.target.len(), 1);
                assert_eq!(draft_trimmed.len(), 1);
                assert_ne!(trimmed.target[0].group, draft_trimmed[0].group);
                for report in trimmed.target.iter().chain(draft_trimmed) {
                    assert!(report.released_payload_bytes <= 1024);
                    assert_eq!(report.remaining.retained_payload_bytes, 0);
                    assert!(report.reclaimed_backing_bytes.value().is_none());
                }
                if std::env::var_os("EREDU_SPECULATIVE_CONTINUATION_MODEL").is_none() {
                    assert!(trimmed.target[0].released_payload_bytes > 0);
                    assert!(draft_trimmed[0].released_payload_bytes > 0);
                }
                let repeated_trim = session.trim_parameter_conversions().unwrap();
                assert_eq!(repeated_trim.target[0].released_claims, 0);
                assert_eq!(repeated_trim.external_drafter.unwrap()[0].released_claims, 0);
                assert_eq!(session.snapshot_usage(), saved_usage);
                assert_eq!(session.token_ids(), tokens);
                assert_eq!(session.sampling_state(), sampling);
                let after_trim = session.forecast_remaining_generation(remaining, &options).unwrap();
                assert_eq!(after_trim.continuation, exchanged.continuation);
                session.exchange(&branch)?;
                session.release_branch(&branch)?;

                let baseline = eredu_backend_mlx::allocator_memory().unwrap().active_bytes();
                let forecast = session.forecast_remaining_generation(remaining, &options).unwrap();
                reset_local_allocator_peak().unwrap();
                let mut rejected_pending = false;
                let mut accepted = 0;
                let mut rejected = 0;
                while let Some(step) = session.step()? {
                    if let Some(verification) = &step.verification {
                        accepted += verification.dispositions.iter().filter(|d| **d == SpeculativeProposalDisposition::Accepted).count();
                        rejected += verification.dispositions.iter().filter(|d| **d == SpeculativeProposalDisposition::Rejected).count();
                    }
                    let status = session.status();
                    let outlook = session.forecast_remaining_generation(4, &options);
                    if !session.can_snapshot() || status == SpeculativeRequestStatus::Completed {
                        assert!(outlook.is_err());
                        rejected_pending |= status != SpeculativeRequestStatus::Completed;
                        if matches!(status, SpeculativeRequestStatus::ReadyToSubmitVerification
                            | SpeculativeRequestStatus::TargetVerificationInFlight) {
                            assert!(matches!(session.trim_parameter_conversions(),
                                Err(ParameterConversionTrimError::NotQuiescent)));
                        }
                    } else {
                        let outlook = outlook.unwrap();
                        assert!(outlook.continuation.target.current_positions >= prompt);
                    }
                }
                assert!(rejected_pending);
                assert!(accepted + rejected > 0);
                eprintln!("Post-trim verification: lookahead={lookahead}, accepted={accepted}, rejected={rejected}");
                let growth = eredu_backend_mlx::allocator_memory().unwrap().peak_bytes().saturating_sub(baseline);
                let upper = forecast.estimate.domains[0].additional_generation_peak.upper_bytes.unwrap();
                assert!(growth <= upper, "growth {growth}, forecast {upper}");
                let committed = session.token_ids().to_vec();
                session.restore(&saved)?;
                let replay_usage = session.snapshot_usage();
                let replay_sampling = session.sampling_state();
                session.trim_parameter_conversions().unwrap();
                assert_eq!(session.snapshot_usage(), replay_usage);
                assert_eq!(session.sampling_state(), replay_sampling);
                let restored = session.forecast_remaining_generation(remaining, &options).unwrap();
                assert_eq!(restored.continuation.target.current_positions, prompt);
                assert_eq!(restored.continuation.draft.as_ref().unwrap().current_positions, prompt);
                session.release_snapshot(&saved)?;
                while session.step()?.is_some() {}
                assert_eq!(session.token_ids(), committed);
                eprintln!("Speculative continuation: model={}, prompt={prompt}, lookahead={lookahead}, horizon={remaining}, upper={upper}, measured_growth={growth}, target_current={:?}, draft_current={:?}",
                    path.display(), forecast.continuation.target.current_state, forecast.continuation.draft.as_ref().unwrap().current_state);
                Ok(())
            },
        ).unwrap();
        assert_eq!(output.token_ids(), expected.token_ids());
        assert_eq!(output.finish_reason(), expected.finish_reason());
    }
}
