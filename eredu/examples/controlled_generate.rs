//! Complete local facade workflow; bounded JSONL records go to stdout.
use eredu::{
    api::{
        local_device_plan, ControlledGenerationRecord, GenerationBranchOptions, LoadedModel,
        LocalDevice, ObservedGenerationEvent, PreparedChatGenerationSettings, SamplingOverride,
        TraceLimits,
    },
    runtime::chat::{ChatTemplateRequest, ToolChoice},
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    capture::*, execution_control::*, intervention::*, ExecutionPlan, GenerationConfigOverrides,
    ObservationSupportStatus, SemanticEvent, SessionCapabilities,
};
use std::ops::ControlFlow;

fn output(
    events: &mut Vec<SemanticEvent>,
) -> impl FnMut(ControlledGenerationRecord) -> ControlFlow<()> + '_ {
    |record| {
        if let ObservedGenerationEvent::Semantic { event, .. } = &record.generation.event {
            events.push(event.clone());
        }
        println!(
            "{}",
            serde_json::to_string(&record).expect("versioned record")
        );
        ControlFlow::Continue(())
    }
}

fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let artifact = arguments.first().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: controlled_generate <artifact-with-tokenizer> [prompt] [f32|f16|bf16]"
        )
    })?;
    let prompt = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("Explain gravity briefly.");
    let dtype = match arguments.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => InterventionDtype::Float32,
        "f16" => InterventionDtype::Float16,
        "bf16" => InterventionDtype::Bfloat16,
        other => anyhow::bail!("unknown activation dtype {other}"),
    };
    run_example(std::path::Path::new(artifact), prompt, dtype)
}

/// Runs the same complete workflow used by the CLI and its native fixture test.
pub fn run_example(
    artifact: &std::path::Path,
    prompt: &str,
    dtype: InterventionDtype,
) -> anyhow::Result<()> {
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu)?)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), artifact, &execution)?
            .into_parts();
    let discovery = model.intervention_discovery()?;
    let target = discovery
        .points
        .iter()
        .find(|point| {
            point.stage == InterventionStage::LogitsBeforeSampling
                && point.decode == ObservationSupportStatus::Supported
                && point.operations.contains(&InterventionKind::Scale)
                && point.operations.contains(&InterventionKind::Zero)
                && point.dtypes.contains(&dtype)
        })
        .ok_or_else(|| anyhow::anyhow!("loaded execution lacks the requested logits intervention"))?
        .path
        .clone();
    let usage = CaptureUsage {
        captures: 128,
        retained_bytes: 256 << 20,
        host_bytes: 4 << 20,
        encoded_bytes: 1 << 20,
    };
    let capture_limits = CaptureLimits {
        per_step: usage,
        cumulative: usage,
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Fail,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "raw-logits".into(),
            path: target.clone(),
            schedule: CaptureSchedule {
                prefill: false,
                ..Default::default()
            },
            slices: vec![],
            transform: CaptureTransform::Preview { max_elements: 8 },
        }],
        limits: capture_limits.clone(),
    };
    let intervention = |action| InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "future-logits".into(),
            target: target.clone(),
            schedule: CaptureSchedule {
                prefill: false,
                ..Default::default()
            },
            slices: vec![],
            action,
            evidence: InterventionEvidence::Preview { max_elements: 8 },
        }],
    };
    let original = intervention(InterventionAction::Scale { dtype, factor: 1.0 });
    let modified = intervention(InterventionAction::Zero { dtype });
    // Current complete-snapshot coverage requires the ordinary forbidden-tool
    // constraint owner. Active/automatic llguidance state has no complete cost
    // estimator. This request declares a tool surface and explicitly forbids calls.
    let chat = model.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":prompt})],
        tools: vec![serde_json::json!({"type":"function", "function": {
            "name":"lookup", "parameters":{"type":"object", "properties":{}, "additionalProperties":false}
        }})], tool_choice: ToolChoice::None, enable_thinking: Some(false),
        add_generation_prompt: true, ..Default::default()
    })?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(12),
            ..Default::default()
        },
        seed: 42,
        ..Default::default()
    };
    let trace = TraceLimits {
        per_record_bytes: 64 << 10,
        total_bytes: 128 << 10,
    };
    let prepared = model.prepare_intervened_chat(&chat, settings, capture, original, trace)?;
    let mut prefix = vec![];
    let mut run = model.start_controlled_chat(
        prepared,
        &[],
        GenerationControlHandle::default(),
        output(&mut prefix),
    )?;
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 512 << 20,
        cumulative_copy_bytes: 2 << 30,
    })?;
    for _ in 0..2 {
        anyhow::ensure!(
            matches!(
                run.status(),
                GenerationStatus::Prepared | GenerationStatus::Paused
            ),
            "generation ended before the fork boundary; use a different prompt"
        );
        run.step(output(&mut prefix))?;
    }
    run.pause(output(&mut prefix))?;
    let snapshot = run.snapshot(output(&mut prefix))?;
    let mut baseline = vec![];
    run.resume(output(&mut baseline))?;
    let baseline_ids = run.token_ids().to_vec();
    let baseline_finish = run.finish_reason();
    run.restore(&snapshot, output(&mut Vec::new()))?;
    let mut restored = vec![];
    run.resume(output(&mut restored))?;
    anyhow::ensure!(
        run.token_ids() == baseline_ids
            && restored == baseline
            && run.finish_reason() == baseline_finish,
        "restored continuation disagreed with the baseline"
    );
    let options = || GenerationBranchOptions {
        trace_limits: trace,
        capture_limits: Some(capture_limits.clone()),
        sampling: None,
        intervention: None,
    };
    let mut branch = run.fork(&snapshot, options(), output(&mut Vec::new()))?;
    run.exchange(&mut branch, output(&mut Vec::new()))?;
    let mut unchanged = vec![];
    run.run(output(&mut unchanged))?;
    anyhow::ensure!(
        run.token_ids() == baseline_ids
            && unchanged == baseline
            && run.finish_reason() == baseline_finish,
        "isolated continuation disagreed with the baseline"
    );
    run.exchange(&mut branch, output(&mut Vec::new()))?;
    drop(branch);
    let mut changes = options();
    changes.sampling = Some(SamplingOverride {
        temperature: Some(0.5),
        reseed: Some(711),
    });
    changes.intervention = Some(modified);
    let mut branch = run.fork(&snapshot, changes, output(&mut Vec::new()))?;
    run.exchange(&mut branch, output(&mut Vec::new()))?;
    run.run(output(&mut Vec::new()))?;
    eprintln!(
        "verified restore and unchanged fork; baseline={baseline_ids:?}; modified={:?}; usage={:?}",
        run.token_ids(),
        run.snapshot_usage()
    );
    Ok(())
}
