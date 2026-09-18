//! Baseline/reset/modified experiment through the ordinary facade generator.
use eredu::api::{
    ChatSourceInput, LoadedModel, LocalDevice, PreparedChatGenerationSettings, PreparedChatRequest,
    TokenizerSourceInput, TraceLimits, local_device_plan,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    ExecutionPlan, GenerationConfigOverrides, ObservationSupportStatus, SessionCapabilities,
    capture::*, intervention::*,
};
use std::ops::ControlFlow;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let path = args.first().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: intervened_generate <artifact-with-tokenizer> [prompt] [f32|f16|bf16]"
        )
    })?;
    let prompt = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("Explain gravity briefly.");
    let dtype = match args.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => InterventionDtype::Float32,
        "f16" => InterventionDtype::Float16,
        "bf16" => InterventionDtype::Bfloat16,
        other => anyhow::bail!("unknown exact activation dtype {other}"),
    };
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu)?)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &execution)?
            .into_parts();
    let discovery = model.intervention_discovery()?;
    eprintln!("{}", serde_json::to_string_pretty(&discovery)?);
    let point = discovery
        .points
        .iter()
        .find(|point| {
            point.stage == InterventionStage::LogitsBeforeSampling
                && point.prefill == ObservationSupportStatus::Supported
                && point.operations.contains(&InterventionKind::Zero)
                && point.dtypes.contains(&dtype)
        })
        .ok_or_else(|| {
            anyhow::anyhow!("no supported prefill logits intervention for requested dtype")
        })?;
    let budget = CaptureUsage {
        captures: 100,
        retained_bytes: 1024 * 1024 * 1024,
        host_bytes: 32 * 1024 * 1024,
        encoded_bytes: 8 * 1024 * 1024,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "original-logits".into(),
            path: point.path.clone(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 8 },
        }],
        limits: CaptureLimits {
            per_step: budget,
            cumulative: budget,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let trace = TraceLimits {
        per_record_bytes: 1024 * 1024,
        total_bytes: 16 * 1024 * 1024,
    };
    const CAPACITY: u64 = 64 << 30;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        inference: eredu_core::TextInferencePolicy {
            managed_memory_capacity_bytes: Some(CAPACITY),
            ..Default::default()
        },
        seed: 42,
        ..Default::default()
    };
    let cancellation = eredu_core::GenerationCancellationToken::new();
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
    let emit = |record: eredu::api::ControlledGenerationRecord| {
        println!(
            "{}",
            serde_json::to_string(&record).expect("versioned host record")
        );
        ControlFlow::Continue(())
    };
    let mut request = PreparedChatRequest::new(&chat, settings);
    request.capture = Some(&capture);
    eprintln!("Baseline");
    let mut run = model
        .start_controlled_chat(request, trace, Default::default(), emit)?
        .ok_or_else(|| anyhow::anyhow!("cancelled before baseline"))?;
    let last_prompt_row = run.prompt_attribution().decoder_positions - 1;
    run.run(emit)?;
    let baseline_ids = run.token_ids().to_vec();
    drop(run);
    model.reset()?;
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "zero-first-prediction-logits".into(),
            target: point.path.clone(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: vec![CaptureSlice {
                axis: "sequence".into(),
                start: last_prompt_row,
                end: last_prompt_row + 1,
                stride: 1,
            }],
            action: InterventionAction::Zero { dtype },
            evidence: InterventionEvidence::Preview { max_elements: 8 },
        }],
    };
    let mut request = PreparedChatRequest::new(&chat, settings);
    request.capture = Some(&capture);
    request.intervention = Some(&plan);
    let mut modified = model
        .start_controlled_chat(request, trace, Default::default(), emit)?
        .ok_or_else(|| anyhow::anyhow!("cancelled before modified run"))?;
    modified.run(emit)?;
    eprintln!(
        "Baseline IDs: {:?}; modified IDs: {:?}",
        baseline_ids,
        modified.token_ids()
    );
    Ok(())
}
