//! Exercise the public request, its automatic native capacities, and borrowed delivery.
//! Each mode owns a fresh process so ordinary initialization cannot supply admission authority.
use super::*;
use eredu_core::{
    FinishReason, GenerationCancellationToken, GenerationPlainTextEvent, TextGenerationConfig,
    TextInferencePolicy,
};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::{error::Error, num::NonZeroU64};

#[path = "managed_plain/lifecycle.rs"]
mod lifecycle;

const CASE: &str = "managed_plain::native_managed_plain_matches_ordinary_and_controlled_generation";
const MODE_ENV: &str = "EREDU_PUBLIC_MANAGED_PLAIN_MODE";
const RESULT_PREFIX: &str = "PUBLIC_MANAGED_PLAIN_RESULT:";
const PROMPT: &str = "abcde";

fn managed_fixture(fixture: Fixture) -> Fixture {
    managed_fixture_with_vocabulary(
        fixture,
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?",
    )
}

fn managed_fixture_with_vocabulary(fixture: Fixture, alphabet: &str) -> Fixture {
    let config_path = fixture.0.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let vocabulary: serde_json::Map<String, serde_json::Value> = alphabet
        .chars()
        .enumerate()
        .map(|(id, character)| (character.to_string(), serde_json::json!(id)))
        .collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "normalizer": null, "pre_tokenizer": null, "post_processor": null,
        "added_tokens": [],
        "decoder": {"type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "vocab": vocabulary, "merges": []}
    });
    std::fs::write(
        fixture.0.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    fixture
}

fn settings(temperature: f32) -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(temperature),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        seed: 827,
        inference: TextInferencePolicy {
            prefill_chunk_positions: NonZeroU64::new(2),
            managed_memory_capacity_bytes: Some(8 * 1024 * 1024 * 1024),
            // Public callers supply one total budget; native component capacities
            // must be derived from the selected execution before admission.
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        },
        ..Default::default()
    }
}

fn budget_failure(error: &(dyn Error + 'static)) -> bool {
    let mut cause = Some(error);
    while let Some(error) = cause {
        if matches!(
            error.downcast_ref::<WorkingMemoryError>(),
            Some(
                WorkingMemoryError::BudgetExceeded { .. }
                    | WorkingMemoryError::CapacityBelowUsage { .. }
            )
        ) {
            return true;
        }
        cause = error.source();
    }
    false
}

fn report_failure<T>(error: impl Error + 'static) -> T {
    let mut cause: Option<&(dyn Error + 'static)> = Some(&error);
    let mut depth = 0;
    while let Some(error) = cause {
        eprintln!("PUBLIC_MANAGED_FAILURE[{depth}]: {error}");
        eprintln!("PUBLIC_MANAGED_FAILURE_DEBUG[{depth}]: {error:?}");
        cause = error.source();
        depth += 1;
    }
    panic!("public managed operation failed; source chain printed above")
}

fn run_mode(mode: &str, fixture: Fixture, temperature: f32) -> serde_json::Value {
    run_mode_with_strategy(
        mode,
        fixture,
        temperature,
        eredu_core::TextSamplingStrategy::Standard,
    )
}

fn run_mode_with_strategy(
    mode: &str,
    fixture: Fixture,
    temperature: f32,
    strategy: eredu_core::TextSamplingStrategy,
) -> serde_json::Value {
    run_mode_with_residency(mode, fixture, temperature, strategy, None)
}

fn run_mode_with_residency(
    mode: &str,
    fixture: Fixture,
    temperature: f32,
    strategy: eredu_core::TextSamplingStrategy,
    residency: Option<eredu_core::ResidencyPlan>,
) -> serde_json::Value {
    run_mode_with_state_residency(
        mode,
        fixture,
        temperature,
        strategy,
        residency,
        eredu_runtime::CacheResidencyPolicy::Device,
    )
}

fn run_mode_with_state_residency(
    mode: &str,
    fixture: Fixture,
    temperature: f32,
    strategy: eredu_core::TextSamplingStrategy,
    residency: Option<eredu_core::ResidencyPlan>,
    state: eredu_runtime::CacheResidencyPolicy,
) -> serde_json::Value {
    run_mode_with_state_residency_observed(
        mode,
        fixture,
        temperature,
        strategy,
        residency,
        state,
        None,
    )
}

fn run_mode_with_state_residency_observed(
    mode: &str,
    fixture: Fixture,
    temperature: f32,
    strategy: eredu_core::TextSamplingStrategy,
    residency: Option<eredu_core::ResidencyPlan>,
    state: eredu_runtime::CacheResidencyPolicy,
    after: Option<&dyn Fn(&LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>)>,
) -> serde_json::Value {
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap());
    let execution = match residency {
        Some(residency) => execution.with_residency(residency),
        None => execution,
    };
    run_mode_with_execution_observed(mode, fixture, temperature, strategy, execution, state, after)
}

fn run_mode_with_execution_observed(
    mode: &str,
    fixture: Fixture,
    temperature: f32,
    strategy: eredu_core::TextSamplingStrategy,
    execution: ExecutionPlan,
    state: eredu_runtime::CacheResidencyPolicy,
    after: Option<&dyn Fn(&LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>)>,
) -> serde_json::Value {
    let root = managed_fixture(fixture);
    let (mut model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default().with_state_residency(state),
        &root.0,
        &execution,
    )
    .unwrap()
    .into_parts();
    let mut settings = settings(temperature);
    settings.strategy = strategy;
    if mode == "ordinary" {
        let prompt = model.encode(PROMPT, false).unwrap();
        assert_eq!(prompt, [0, 1, 2, 3, 4]);
        let config = model.resolve_generation_config(settings.overrides).unwrap();
        let config = TextGenerationConfig::new(config).with_seed(settings.seed);
        let config = match strategy {
            eredu_core::TextSamplingStrategy::Standard => config,
            eredu_core::TextSamplingStrategy::MirostatV2 { tau, eta } => {
                config.with_mirostat_v2(tau, eta).unwrap()
            }
        };
        let ids: Vec<u32> = model
            .generate_tokens(prompt, config)
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        assert_eq!(ids.len(), 4);
        let text = model.decode(&ids, true).unwrap();
        if let Some(after) = after {
            after(&model);
        }
        return serde_json::json!({"ids": ids, "text": text});
    }
    assert!(matches!(mode, "managed" | "controlled"));
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(root.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap();
    let request = ManagedPlainTextRequest::new(PROMPT, settings);
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    assert!(
        model
            .start_managed_plain_text(&source, request, &cancelled)
            .unwrap()
            .is_none()
    );
    let cancellation = GenerationCancellationToken::new();
    let mut short = request;
    short.settings.inference.managed_memory_capacity_bytes = Some(1);
    let refusal = match model.start_managed_plain_text(&source, short, &cancellation) {
        Err(error) => error,
        Ok(_) => panic!("one byte cannot admit the public request"),
    };
    assert!(budget_failure(&refusal), "{refusal:?}");
    drop(refusal);
    let mut visible = String::new();
    let mut finishes = Vec::new();
    let mut emit = |event: GenerationPlainTextEvent<'_>| match event {
        GenerationPlainTextEvent::TextDelta(text) => visible.push_str(text),
        GenerationPlainTextEvent::Finished { reason } => finishes.push(reason),
    };
    let output = if mode == "controlled" {
        let mut session = model
            .start_managed_plain_text(&source, request, &cancellation)
            .unwrap_or_else(report_failure)
            .expect("live request");
        while session.finish_reason().is_none() {
            session = session.advance(&cancellation, &mut emit).unwrap();
        }
        session
            .into_output()
            .unwrap_or_else(|_| panic!("terminal session"))
    } else {
        model
            .generate_managed_plain_text(&source, request, &cancellation, &mut emit)
            .unwrap_or_else(report_failure)
            .expect("live request")
    };
    assert_eq!(output.finish_reason, FinishReason::MaxTokens);
    assert_eq!(finishes, [FinishReason::MaxTokens]);
    assert_eq!(output.token_ids.as_ref().len(), 4);
    assert_eq!(output.text.as_str(), visible);
    if let Some(after) = after {
        after(&model);
    }
    // The public output retains its actual storage after the model and source retire.
    let text_address = output.text.as_str().as_ptr();
    drop((source, model));
    assert_eq!(output.text.as_str().as_ptr(), text_address);
    serde_json::json!({"ids": output.token_ids.as_ref(), "text": output.text.as_str()})
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_plain_matches_ordinary_and_controlled_generation() {
    verify_modes(CASE, MODE_ENV, 0.0, || fixture(false));
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_hybrid_matches_ordinary_and_controlled_generation() {
    // Reuse the nonzero mixed recurrent/full-attention checkpoint and the same
    // public request driver. Five prompt tokens exercise chunks 2/2/1, then
    // four decode steps reuse both recurrent and KV state.
    verify_modes(
        "managed_plain::native_managed_hybrid_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_HYBRID_MODE",
        0.0,
        super::components::qwen_35_fixture,
    );
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_standard_sampling_matches_ordinary_and_controlled_generation() {
    // Positive temperature retains the actual default filters and seed827;
    // every mode advances the same explicit native key and categorical worker.
    verify_modes(
        "managed_plain::native_managed_standard_sampling_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_STANDARD_MODE",
        1.0,
        || fixture(false),
    );
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_mirostat_matches_ordinary_and_controlled_generation() {
    verify_modes_with_strategy(
        "managed_plain::native_managed_mirostat_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_MIROSTAT_MODE",
        1.0,
        || fixture(false),
        eredu_core::TextSamplingStrategy::MirostatV2 { tau: 5.0, eta: 0.1 },
    );
}

fn verify_modes(case: &str, mode_env: &str, temperature: f32, make_fixture: fn() -> Fixture) {
    verify_modes_with_strategy(
        case,
        mode_env,
        temperature,
        make_fixture,
        eredu_core::TextSamplingStrategy::Standard,
    );
}

fn verify_modes_with_strategy(
    case: &str,
    mode_env: &str,
    temperature: f32,
    make_fixture: fn() -> Fixture,
    strategy: eredu_core::TextSamplingStrategy,
) {
    verify_modes_with_residency(case, mode_env, temperature, make_fixture, strategy, None)
}

fn verify_modes_with_residency(
    case: &str,
    mode_env: &str,
    temperature: f32,
    make_fixture: fn() -> Fixture,
    strategy: eredu_core::TextSamplingStrategy,
    residency: Option<eredu_core::ResidencyPlan>,
) {
    verify_mode_results(case, mode_env, |mode| {
        if residency.is_some() {
            run_mode_with_residency(&mode, make_fixture(), temperature, strategy, residency)
        } else {
            match strategy {
                eredu_core::TextSamplingStrategy::Standard => {
                    run_mode(&mode, make_fixture(), temperature)
                }
                _ => run_mode_with_strategy(&mode, make_fixture(), temperature, strategy),
            }
        }
    });
}
fn verify_mode_results(case: &str, mode_env: &str, run: impl FnOnce(&str) -> serde_json::Value) {
    verify_mode_results_with_reference(case, mode_env, |mode, _| run(mode));
}
fn verify_mode_results_with_reference(
    case: &str,
    mode_env: &str,
    run: impl FnOnce(&str, Option<serde_json::Value>) -> serde_json::Value,
) {
    let reference_env = format!("{mode_env}_REFERENCE");
    if let Ok(mode) = std::env::var(mode_env) {
        let reference = std::env::var(&reference_env).ok()
            .map(|value| serde_json::from_str(&value).unwrap());
        let result = run(&mode, reference);
        println!("\n{RESULT_PREFIX}{result}");
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["--exact", case, "--ignored", "--nocapture"]).env(mode_env, mode);
        if let Some(reference) = &expected {
            command.env(&reference_env, serde_json::to_string(reference).unwrap());
        } else {
            command.env_remove(&reference_env);
        }
        let result = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(RESULT_PREFIX))
                .unwrap_or_else(|| panic!("missing positive execution marker: {mode}: {stdout}")),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{mode}");
        } else {
            expected = Some(actual);
        }
    }
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_host_layerwise_matches_ordinary_and_controlled_generation() {
    // Two nonzero decoder units and a one-unit device window exercise actual
    // host copies between every prefill/decode unit, through the public driver.
    verify_modes_with_residency(
        "managed_plain::native_managed_host_layerwise_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_HOST_LAYERWISE_MODE",
        0.0,
        || fixture(false),
        eredu_core::TextSamplingStrategy::Standard,
        Some(eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        }),
    );
}

#[test]
#[ignore = "requires an accessible Metal device and complete managed native qualification"]
fn native_managed_foreground_disk_matches_ordinary_and_controlled_generation() {
    // Three nonzero units exceed the selected two-unit device window. Zero
    // host capacity exercises synchronous source reads and repeated eviction.
    verify_modes_with_residency(
        "managed_plain::native_managed_foreground_disk_matches_ordinary_and_controlled_generation",
        "EREDU_PUBLIC_MANAGED_FOREGROUND_DISK_MODE",
        0.0,
        || {
            let fixture = fixture(false);
            let path = fixture.0.join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            config["num_hidden_layers"] = 3.into();
            std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
            let resolved =
                eredu_architectures::configuration::resolve_model_config(&config).unwrap();
            write_tensor_plan(&fixture.0, resolved.architecture.checkpoint());
            fixture
        },
        eredu_core::TextSamplingStrategy::Standard,
        Some(eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 0,
            host_lookahead: 0,
            background_queue: 0,
        }),
    );
}

#[path = "managed_plain/snapshot.rs"]
mod snapshot;

#[path = "managed_plain/speculative.rs"]
mod speculative;

#[path = "managed_plain/capture.rs"]
mod capture;

#[path = "managed_plain/embedded.rs"]
mod embedded;

#[path = "managed_plain/background.rs"]
mod background;

#[path = "managed_plain/paged.rs"]
mod paged;

#[path = "managed_plain/parallel.rs"]
mod parallel;

#[cfg(all(feature = "metal", feature = "image", feature = "audio"))]
#[path = "managed_plain/original_media.rs"]
mod original_media;

#[path = "managed_plain/expert_cache.rs"]
mod expert_cache;
