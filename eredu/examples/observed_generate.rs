//! Complete facade workflow: cold discovery, bounded admission, ordinary sampling,
//! attributed streaming, and cancellation. No native tensors or custom generation loop.
use eredu::api::{
    ChatSourceInput, LoadedModel, LocalDevice, ObservedGenerationEvent,
    PreparedChatGenerationSettings, PreparedChatRequest, TokenizerSourceInput, TraceLimits,
    inspect_architecture, local_device_plan,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides,
    ObservationSupportStatus, SessionCapabilities, capture::*,
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
    const CAPACITY: u64 = 64 << 30;
    let cancellation = GenerationCancellationToken::new();
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancellation,
        )?
        .ok_or_else(|| anyhow::anyhow!("cancelled before chat source"))?;
    let policy = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role": "user", "content": prompt})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, CAPACITY, &cancellation)?
        .ok_or_else(|| anyhow::anyhow!("cancelled before chat preparation"))?;
    let mut request = PreparedChatRequest::new(
        &chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.7),
                max_new_tokens: Some(32),
                ..Default::default()
            },
            inference: eredu_core::TextInferencePolicy {
                managed_memory_capacity_bytes: Some(CAPACITY),
                ..Default::default()
            },
            seed: 42,
            ..Default::default()
        },
    );
    request.capture = Some(&plan);
    let trace = TraceLimits {
        per_record_bytes: 2 * 1024 * 1024,
        total_bytes: 16 * 1024 * 1024,
    };
    let control = eredu_core::execution_control::GenerationControlHandle::default();
    let cancel = control.clone();
    let mut emit = |record: eredu::api::ControlledGenerationRecord| {
        println!(
            "{}",
            serde_json::to_string(&record).expect("versioned host record")
        );
        if matches!(record.event.progress(), Some(ObservedGenerationEvent::Token { prediction_index, .. }) if prediction_index + 1 >= cancel_after)
        {
            cancel.cancel();
        }
        ControlFlow::Continue(())
    };
    let mut session = model
        .start_controlled_chat(request, trace, control, &mut emit)?
        .ok_or_else(|| anyhow::anyhow!("cancelled before generation"))?;
    eprintln!(
        "Admitted {} prompt positions",
        session.prompt_attribution().decoder_positions
    );
    session.run(&mut emit)?;
    eprintln!(
        "Stopped: {:?}; {} generated tokens",
        session.finish_reason(),
        session.token_ids().len()
    );
    Ok(())
}
