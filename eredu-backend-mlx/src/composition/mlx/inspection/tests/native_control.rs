use super::*;
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod family_residency;
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod paired_copy;
use eredu_core::{ModelRuntime, TextGenerationBackend};

type Backend = crate::backend::MlxBackend<'static>;

#[derive(Clone, Debug, Default, PartialEq)]
struct SnapshotController {
    history: [u32; 20],
    committed: usize,
}
impl eredu_runtime::execution_control::SnapshotTokenController for SnapshotController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        Some(std::mem::size_of::<Self>() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        self.snapshot_storage_bytes()
    }
    fn fork_original_snapshot(&self) -> Option<Self> {
        Some(self.clone())
    }
}
impl eredu_core::TokenFilterController for SnapshotController {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: eredu_core::TextFilterWorkspace::OptionalMask {
                max_mask_positions: 64,
                mask_capacity_bytes: 64,
            },
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        // A changing canonical allow mask makes lost controller state observable.
        let mut allowed = vec![true; 64];
        allowed[self.committed % 64] = false;
        Ok(eredu_core::TokenFilter::allowed(allowed).unwrap())
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.history[self.committed] = token;
        self.committed += 1;
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

#[test]
fn native_control_complete_continuation_preserves_rng_adaptation_constraints_and_capture() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_values(root.path(), |key, index| {
        if key.contains("norm") {
            1.0
        } else {
            (((index * 17 + key.len() * 7) % 101) as f32 - 50.0) * 0.003
        }
    });
    sampled_conformance(root.path(), false);
}

#[test]
fn native_control_heterogeneous_convolution_continuations_are_isolated() {
    let root = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
        "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
    });
    write_configured_safetensors_fixture(root.path(), &config, fixture_value);
    sampled_conformance(root.path(), false);
}

#[test]
fn native_control_recurrent_moe_continuations_preserve_routing_interventions() {
    let root = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "model_type": "qwen3_5_moe_text", "vocab_size": 64, "hidden_size": 32,
        "num_hidden_layers": 2, "mtp_num_hidden_layers": 0,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "intermediate_size": 48, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24, "num_experts_per_tok": 1,
        "num_experts": 2, "layer_types": ["linear_attention", "full_attention"]
    });
    write_configured_safetensors_fixture(root.path(), &config, fixture_value);
    sampled_conformance(root.path(), true);
}

fn fixture_value(key: &str, index: usize) -> f32 {
    if key.contains("norm") {
        1.0
    } else {
        (((index * 17 + key.len() * 7) % 101) as f32 - 50.0) * 0.003
    }
}

fn sampled_conformance(root: &Path, routed: bool) {
    if !crate::tests::support::native_process::enter("prepared-native-continuation") {
        return;
    }
    sampled_conformance_using(routed, || load(root), |_| {});
}

fn sampled_conformance_using(
    routed: bool,
    load: impl Fn() -> ModelRuntime<Backend>,
    inspect: impl Fn(&ModelRuntime<Backend>),
) {
    use eredu_core::{
        capture::*, intervention::*, GenerationConfigOverrides, TextGenerationConfig,
    };
    use eredu_evaluation::execution_control::{
        continuation_conformance, forced_choice_conformance, sampling_override_conformance,
        ContinuationFixtureLimits,
    };
    for adaptive in [false, true] {
        // Ordinary, capture-only, intervention-only and combined shared owners.
        for mode in 0..4 {
            let mut runtime = load();
            let discovery = Backend::capture_discovery(&runtime).unwrap();
            let intervention_discovery = Backend::intervention_discovery(&runtime).unwrap();
            let usage = CaptureUsage {
                captures: 10_000,
                retained_bytes: 32_000_000,
                host_bytes: 32_000_000,
                encoded_bytes: 32_000_000,
            };
            let request = CaptureRequestShape {
                batch: 1,
                prompt_tokens: 2,
                max_predictions: 20,
            };
            let capture = CapturePlan {
                schema_version: 1,
                selections: if mode & 1 != 0 {
                    vec![CaptureSelection {
                        id: "raw-logits".into(),
                        path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                        schedule: CaptureSchedule::default(),
                        slices: vec![],
                        transform: CaptureTransform::TopCandidates { count: 4 },
                    }]
                } else {
                    vec![]
                },
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            }
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                request,
            )
            .unwrap();
            let mut operations = if mode & 2 != 0 {
                vec![InterventionOperation {
                    id: "future-scale".into(),
                    target: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                    schedule: CaptureSchedule {
                        first_prediction: 2,
                        every: 2,
                        prefill: false,
                        ..Default::default()
                    },
                    slices: vec![],
                    action: InterventionAction::Scale {
                        dtype: InterventionDtype::Float32,
                        factor: 0.8,
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 2 },
                }]
            } else {
                vec![]
            };
            if routed && mode & 2 != 0 {
                let (point, action) = match intervention_discovery
                    .points
                    .iter()
                    .find(|p| p.routing.is_some())
                {
                    Some(point) => (
                        point,
                        InterventionAction::ForceExperts {
                            shape: [1, 1],
                            expert_ids: vec![1],
                        },
                    ),
                    None => {
                        let point = intervention_discovery
                            .points
                            .iter()
                            .find(|p| p.routed_units.is_some())
                            .expect("routed fixture must expose an actual expert intervention");
                        (
                            point,
                            InterventionAction::Scale {
                                dtype: InterventionDtype::Float32,
                                factor: 0.7,
                            },
                        )
                    }
                };
                operations.push(InterventionOperation {
                    id: "future-route".into(),
                    target: point.path.clone(),
                    schedule: CaptureSchedule {
                        first_prediction: 2,
                        every: 2,
                        prefill: false,
                        ..Default::default()
                    },
                    slices: vec![],
                    action,
                    evidence: if point.routed_units.is_some() {
                        // Sparse unit evidence belongs to an attributed RoutedUnits
                        // capture; this intervention retains its application outcome.
                        InterventionEvidence::None
                    } else {
                        InterventionEvidence::Preview { max_elements: 2 }
                    },
                });
            }
            let intervention = InterventionPlan {
                schema_version: 1,
                operations,
            }
            .admit(&intervention_discovery, request, "native-snapshot-root")
            .unwrap();
            let config = TextGenerationConfig::new(
                eredu_core::resolve_generation_config(
                    None,
                    GenerationConfigOverrides {
                        max_new_tokens: Some(20),
                        temperature: Some(0.8),
                        top_k: Some(64),
                        top_p: Some(1.0),
                        min_p: Some(0.0),
                        repetition_penalty: Some(1.15),
                        frequency_penalty: Some(0.05),
                        presence_penalty: Some(0.05),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .with_seed(12345);
            let config = if adaptive {
                config.with_mirostat_v2(5.0, 0.3).unwrap()
            } else {
                config
            };
            let pool = runtime.backend().memory_ledger().clone();
            let options = eredu_core::TextPreparationOptions {
                capture: Some(SharedCapturePlan::new(capture)),
                interventions: Some(
                    pool.compile_intervention_source(
                        PreparedInterventionPlanCopy::inspect(&intervention).unwrap(),
                    )
                    .unwrap()
                    .plan()
                    .clone(),
                ),
            };
            let (controller, semantic) = crate::tests::support::plain_controller::new(
                &pool,
                &runtime.session().fixture_execution_identity(),
                20,
            );
            let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
                crate::tests::support::original_snapshot::NativeSnapshotProvider,
                crate::backend::error::Error,
                crate::backend::error::Error,
            >()
            .unwrap();
            let mut state = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
                &mut runtime,
                eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
                config.clone(),
                eredu_runtime::execution_control::TokenChoiceController::new(
                    SnapshotController::default(),
                    eredu_runtime::TokenDomain::new(64),
                ),
                Some(eredu_core::TextPreparationOptions {
                    capture: options.capture.clone(),
                    interventions: options.interventions.clone(),
                }),
                eredu_core::GenerationSequenceRequest::new(20, &[]).with_consumer(&consumer),
            )
            .unwrap_or_else(|cause| {
                panic!("canonical snapshot startup mode={mode} config={config:?}: {cause:?}")
            });
            let sequence = state
                .take_prepared_sequence()
                .unwrap()
                .prepare_storage()
                .unwrap();
            let mut provider =
                crate::tests::support::original_snapshot::NativeSnapshotProvider::new(
                    sequence,
                    config.clone(),
                    pool.clone(),
                );
            let limits = ContinuationFixtureLimits {
                host_bytes: 4096,
                // The fixture stays within its current 256-token KV allocation;
                // one MiB also bounds its remaining scalar/history growth.
                growth_bytes: 1_048_576,
                max_predictions: 20,
                capture: Some(CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    on_limit: CaptureLimitPolicy::Fail,
                }),
            };
            let probe = || {
                (
                    crate::tests::support::path_instrumentation::snapshot(),
                    crate::tests::support::path_instrumentation::session_reset_attempts(),
                )
            };
            sampling_override_conformance(&mut state, &mut provider, &limits, probe);
            let retirement = continuation_conformance(
                state,
                provider,
                ContinuationFixtureLimits {
                    host_bytes: limits.host_bytes,
                    growth_bytes: limits.growth_bytes,
                    max_predictions: limits.max_predictions,
                    capture: limits.capture.clone(),
                },
                probe,
            );
            runtime.reset().unwrap_or_else(|cause| {
                panic!(
                    "settled continuation reset: {cause:?}; {:?}",
                    pool.snapshot().unwrap()
                )
            });
            retirement.finish();
            let consumer = eredu_core::GenerationSequenceConsumerLayout::for_driver_types::<
                crate::tests::support::original_snapshot::NativeSemanticSnapshotProvider,
                crate::backend::error::Error,
                crate::backend::error::Error,
            >()
            .unwrap();
            let mut forced = eredu_core::ControlledTextGeneration::from_token_ids_with_sequence(
                &mut runtime,
                eredu_core::TokenIdsInputPlan::new(&[1, 2]).unwrap(),
                config.clone(),
                eredu_runtime::execution_control::TokenChoiceController::new(
                    controller,
                    eredu_runtime::TokenDomain::new(64),
                ),
                Some(options),
                eredu_core::GenerationSequenceRequest::new(20, &[])
                    .with_consumer(&consumer)
                    .with_semantic_state(&semantic),
            )
            .unwrap();
            let sequence = forced
                .take_prepared_sequence()
                .unwrap()
                .prepare_storage()
                .unwrap();
            let mut provider =
                crate::tests::support::original_snapshot::NativeSemanticSnapshotProvider::new(
                    sequence, semantic, config, pool,
                );
            forced_choice_conformance(&mut forced, &mut provider, &limits, 64, probe);
            drop((forced, provider));
            inspect(&runtime);
        }
    }
}

fn load(root: &Path) -> ModelRuntime<Backend> {
    let backend = prepared_backend();
    let model = eredu_core::load_model(&backend, root, MlxLoadRequest::default()).unwrap();
    ModelRuntime::from_prepared(backend, model).unwrap()
}
