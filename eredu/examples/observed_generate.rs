//! Complete facade workflow: cold discovery, bounded admission, ordinary sampling,
//! attributed streaming, and cancellation. No native tensors or custom generation loop.
use eredu::api::{
    inspect_architecture, local_device_plan, LoadedModel, LocalDevice, ObservedGenerationEvent,
    PreparedChatGenerationSettings, TraceLimits,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    capture::*, ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides,
    ObservationSupportStatus, SessionCapabilities,
};
use std::ops::ControlFlow;

fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let path = arguments.first().ok_or_else(|| anyhow::anyhow!(
        "usage: observed_generate <artifact-with-tokenizer-and-chat-template> [prompt] [cancel-after-tokens]"))?;
    let prompt = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("Explain why the sky is blue.");
    let cancel_after = arguments
        .get(2)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(8);
    let descriptor = inspect_architecture(path)?;
    eprintln!(
        "Discovered {} observation points without loading weights",
        descriptor.observations.points.len()
    );

    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu)?)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &execution)?
            .into_parts();
    let discovery = model.capture_discovery()?;
    let point = discovery
        .catalog
        .points
        .iter()
        .find(|point| {
            point.path != eredu_core::MODEL_LOGITS_OBSERVATION_PATH
                && discovery.support.points.iter().any(|support| {
                    support.path == point.path
                        && support.prefill == ObservationSupportStatus::Supported
                        && support.decode == ObservationSupportStatus::Supported
                })
        })
        .ok_or_else(|| {
            anyhow::anyhow!("no activation supports both ordinary prefill and decode")
        })?;
    anyhow::ensure!(
        descriptor.observations.get(&point.path).is_some(),
        "loaded catalog differs from cold inspection"
    );
    let per_step = CaptureUsage {
        captures: 2,
        retained_bytes: 64 * 1024 * 1024,
        host_bytes: 1024 * 1024,
        encoded_bytes: 64 * 1024,
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: [
            CaptureTransform::Summary,
            CaptureTransform::Preview { max_elements: 16 },
        ]
        .into_iter()
        .enumerate()
        .map(|(index, transform)| CaptureSelection {
            id: format!("activation-{index}"),
            path: point.path.clone(),
            schedule: CaptureSchedule::default(),
            slices: Vec::new(),
            transform,
        })
        .collect(),
        limits: CaptureLimits {
            per_step,
            cumulative: CaptureUsage {
                captures: 64,
                retained_bytes: 1024 * 1024 * 1024,
                host_bytes: 32 * 1024 * 1024,
                encoded_bytes: 2 * 1024 * 1024,
            },
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Skip,
        },
    };
    let chat = model.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({"role": "user", "content": prompt})],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let prepared = model.prepare_observed_chat(
        &chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(32),
                ..Default::default()
            },
            seed: 42,
            ..Default::default()
        },
        plan,
        TraceLimits {
            per_record_bytes: 2 * 1024 * 1024,
            total_bytes: 16 * 1024 * 1024,
        },
    )?;
    eprintln!(
        "Admitted {} for {} prompt tokens",
        prepared.capture_plan().identity(),
        prepared.prompt_token_ids().len()
    );
    let cancellation = GenerationCancellationToken::new();
    let outcome = model.generate_observed_chat(prepared, &[], cancellation.clone(), |record| {
        println!("{}", serde_json::to_string(&record).expect("versioned host record"));
        if matches!(record.event, ObservedGenerationEvent::Token { prediction_index, .. } if prediction_index + 1 >= cancel_after) {
            // Equivalently, return ControlFlow::Break(()). Delivery is synchronous
            // and the ordinary generator settles work before returning.
            cancellation.cancel();
        }
        ControlFlow::Continue(())
    })?;
    eprintln!(
        "Stopped: {:?}; {} generated tokens",
        outcome.finish_reason,
        outcome.token_ids.len()
    );
    Ok(())
}
