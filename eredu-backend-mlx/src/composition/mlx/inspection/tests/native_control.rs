use super::*;
use eredu_core::{
    execution_control::{
        ControlSupport, NativeTextStateBackend, SnapshotLimits, SnapshotResourceKind,
    },
    Completion, ModelRuntime, TextGenerationBackend,
};
use eredu_runtime::execution_control::{SnapshotBudget, SnapshotReservation};
use safemlx::{Array, Stream};

type Backend = crate::backend::MlxBackend<'static>;

#[derive(Clone, Debug, Default, PartialEq)]
struct SnapshotController {
    history: Vec<u32>,
}
impl eredu_runtime::execution_control::SnapshotTokenController for SnapshotController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add(4 * self.history.len() as u64)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
}
impl eredu_core::TokenFilterController for SnapshotController {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        // A changing canonical allow mask makes lost controller state observable.
        let mut allowed = vec![true; 64];
        allowed[self.history.len() % 64] = false;
        Ok(eredu_core::TokenFilter::allowed(allowed).unwrap())
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.history.push(token);
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
    use eredu_core::{
        capture::*, intervention::*, GenerationConfigOverrides, TextGenerationConfig,
        TextGenerationDriver,
    };
    use eredu_evaluation::execution_control::{
        continuation_conformance, forced_choice_conformance, sampling_override_conformance,
        ContinuationFixtureLimits,
    };
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    for adaptive in [false, true] {
        // Ordinary, capture-only, intervention-only and combined shared owners.
        for mode in 0..4 {
            let mut runtime = load(root, &stream);
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
                    physical_native_bytes: None,
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
                let point = intervention_discovery
                    .points
                    .iter()
                    .find(|p| p.routing.is_some())
                    .expect("fixture must expose real routing");
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
                    action: InterventionAction::ForceExperts {
                        shape: [1, 1],
                        expert_ids: vec![1],
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 2 },
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
            let prompt = Backend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let mut state = driver
                .start(
                    prompt,
                    config,
                    eredu_runtime::execution_control::TokenChoiceController::new(
                        SnapshotController::default(),
                        eredu_runtime::TokenDomain::new(64),
                    ),
                )
                .unwrap();
            driver
                .enable_interventions(&mut state, capture, intervention)
                .unwrap();
            let mut state = eredu_runtime::execution_control::ManagedTextContinuation::root(state);
            let limits = ContinuationFixtureLimits {
                host_bytes: 4096,
                // The fixture stays within its current 256-token KV allocation;
                // one MiB also bounds its remaining scalar/history growth.
                growth_bytes: 1_048_576,
                max_predictions: 20,
                capture: Some(CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                }),
            };
            let probe = || {
                (
                    crate::tests::support::path_instrumentation::snapshot(),
                    crate::tests::support::path_instrumentation::session_reset_attempts(),
                )
            };
            forced_choice_conformance(&mut driver, &mut state, &limits, 64, probe);
            sampling_override_conformance(&mut driver, &mut state, &limits, probe);
            continuation_conformance(&mut driver, state, limits, probe);
        }
    }
}

struct SavedState {
    state: crate::native::MlxNativeTextState,
    _reservation: SnapshotReservation,
}

fn capture(runtime: &mut ModelRuntime<Backend>, budget: &SnapshotBudget) -> SavedState {
    let estimate = Backend::estimate_native_text_state(runtime, None).unwrap();
    let reservation = budget
        .reserve(SnapshotResourceKind::Snapshot, estimate)
        .unwrap();
    SavedState {
        state: Backend::capture_native_text_state(runtime).unwrap(),
        _reservation: reservation,
    }
}

fn copy(
    runtime: &mut ModelRuntime<Backend>,
    budget: &SnapshotBudget,
    saved: &SavedState,
) -> SavedState {
    let estimate = Backend::estimate_native_text_state(runtime, Some(&saved.state)).unwrap();
    let reservation = budget
        .reserve(SnapshotResourceKind::Branch, estimate)
        .unwrap();
    SavedState {
        state: Backend::copy_native_text_state(runtime, &saved.state).unwrap(),
        _reservation: reservation,
    }
}

fn decode(runtime: &mut ModelRuntime<Backend>, token: u32) -> Vec<f32> {
    let submission = runtime
        .decode(Array::from_slice(&[token], &[1, 1]))
        .unwrap();
    submission.completion.wait().unwrap();
    submission
        .output
        .into_logits()
        .unwrap()
        .into_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn load(root: &Path, stream: &Stream) -> ModelRuntime<Backend> {
    let backend = crate::native::backend(stream, stream);
    let model = eredu_core::load_model(&backend, root, MlxLoadRequest::default()).unwrap();
    ModelRuntime::from_prepared(backend, model).unwrap()
}

#[test]
fn native_control_loaded_dense_restores_and_exchanges_without_replay() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture_with_values(root.path(), |key, index| {
        if key.contains("norm") {
            return 1.0;
        }
        let salt = key.bytes().fold(17u32, |n, byte| {
            n.wrapping_mul(31).wrapping_add(u32::from(byte))
        });
        let value = salt
            .wrapping_add(index as u32)
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        ((value >> 8) as f32 / 16777216.0 - 0.5) * 0.3
    });
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let mut runtime = load(root.path(), &stream);
    let mut foreign = load(root.path(), &stream);
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 4,
        max_branches: 4,
        retained_bytes: 32_000_000,
        cumulative_copy_bytes: 128_000_000,
    });
    assert_eq!(
        Backend::native_text_state_support(&runtime),
        ControlSupport::Supported
    );
    let initial = capture(&mut runtime, &budget);
    let prompt = Backend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
    let submission = runtime.prefill(prompt).unwrap();
    // A returned host handle is not completion evidence. Rejection is read-only.
    assert!(Backend::capture_native_text_state(&mut runtime).is_err());
    submission.completion.wait().unwrap();
    let boundary = capture(&mut runtime, &budget);

    let before = crate::tests::support::path_instrumentation::snapshot();
    let resets = crate::tests::support::path_instrumentation::session_reset_attempts();
    let mut sibling = copy(&mut runtime, &budget, &boundary);
    let mut reusable = copy(&mut runtime, &budget, &boundary);
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        before
    );
    assert!(Backend::exchange_native_text_state(&mut foreign, &mut sibling.state).is_err());
    assert!(Backend::validate_native_text_state(&foreign, &boundary.state).is_err());
    // An incompatible-state error does not poison either correctly paired run.
    let foreign_prompt = Backend::prepare_text_prompt(foreign.backend(), vec![5, 6]).unwrap();
    foreign
        .prefill(foreign_prompt)
        .unwrap()
        .completion
        .wait()
        .unwrap();

    let baseline = [3, 4, 5].map(|token| decode(&mut runtime, token));
    let before_switch = crate::tests::support::path_instrumentation::snapshot();
    Backend::exchange_native_text_state(&mut runtime, &mut sibling.state).unwrap();
    assert_eq!(
        crate::tests::support::path_instrumentation::snapshot(),
        before_switch
    );
    assert_eq!(decode(&mut runtime, 3), baseline[0]);
    // Interleave the advanced parent and child. Each slot carries cache positions.
    Backend::exchange_native_text_state(&mut runtime, &mut sibling.state).unwrap();
    let parent_next = decode(&mut runtime, 6);
    Backend::exchange_native_text_state(&mut runtime, &mut sibling.state).unwrap();
    assert_eq!(decode(&mut runtime, 4), baseline[1]);
    assert_eq!(decode(&mut runtime, 5), baseline[2]);
    assert_eq!(decode(&mut runtime, 6), parent_next);

    Backend::exchange_native_text_state(&mut runtime, &mut reusable.state).unwrap();
    let changed = decode(&mut runtime, 19);
    assert_ne!(
        changed, baseline[0],
        "fixture must distinguish alternative model inputs"
    );
    let mut again = copy(&mut runtime, &budget, &boundary);
    Backend::exchange_native_text_state(&mut runtime, &mut again.state).unwrap();
    assert_eq!(decode(&mut runtime, 3), baseline[0]);
    assert_eq!(
        crate::tests::support::path_instrumentation::session_reset_attempts(),
        resets
    );
    let after = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(after.materializations, before.materializations);
    assert_eq!(after.payload_opens, before.payload_opens);
    assert_eq!(
        after.architecture_constructions,
        before.architecture_constructions
    );
    // Exactly the explicit foreign prefill and ten decode submissions above.
    assert_eq!(after.forwards - before.forwards, 11);

    drop((sibling, reusable, again));
    let mut fresh = copy(&mut runtime, &budget, &initial);
    Backend::exchange_native_text_state(&mut runtime, &mut fresh.state).unwrap();
    let prompt = Backend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
    runtime.prefill(prompt).unwrap().completion.wait().unwrap();
    assert_eq!(decode(&mut runtime, 3), baseline[0]);
}
