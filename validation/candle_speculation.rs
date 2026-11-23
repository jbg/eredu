//! Released Candle acceptance for one tokenizer-compatible target and draft.
use eredu::api::{
    inspect_artifact, ArtifactSource, GenerationControlOptions, GenerationRequest, OutputMode,
    PreparedChatGenerationSettings, PreparedChatInput, PreparedTextModel, TextModelOptions,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_candle::CandleBackendFactory;
use eredu_core::{
    capture::{
        CaptureLimitPolicy, CaptureLimits, CaptureOutcome, CapturePlan, CaptureProvenance,
        CaptureSelection, CaptureTransform, CaptureUsage, CAPTURE_SCHEMA_VERSION,
    },
    decoding::{DecodingMode, SpeculationOptions, SpeculationOptionsRequest},
    execution_control::SnapshotLimits,
    DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationConfigOverrides,
    ResidencyPlan, SpeculativeDraft,
};
use std::num::NonZeroUsize;

fn capture_plan() -> CapturePlan {
    let per_step = CaptureUsage {
        captures: 16,
        retained_bytes: 16 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 1 << 20,
    };
    CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        provenance: CaptureProvenance::Session,
        selections: vec![
            CaptureSelection {
                id: "logits".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: Default::default(),
                slices: vec![],
                transform: CaptureTransform::TopCandidates { count: 8 },
            },
            CaptureSelection {
                id: "token".into(),
                path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                schedule: Default::default(),
                slices: vec![],
                transform: CaptureTransform::TokenScores {
                    position: None,
                    token_ids: vec![6403],
                },
            },
        ],
        limits: CaptureLimits {
            per_step,
            cumulative: CaptureUsage {
                captures: 1024,
                retained_bytes: 2 << 30,
                host_bytes: 1 << 30,
                encoded_bytes: 64 << 20,
            },
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut paths = std::env::args_os().skip(1);
    let target = inspect_artifact(&ArtifactSource::new(paths.next().unwrap()))?;
    let draft_path = paths
        .next()
        .unwrap()
        .into_string()
        .expect("draft path is UTF-8");
    let target_device = paths.next().map(|device| device.into_string().expect("UTF-8 device"));
    let device = DevicePlan::new("candle", target_device.as_deref().unwrap_or("cpu:0"))?;
    let placement = match paths.next() {
        Some(draft_device) => DraftPlacementPlan::Device {
            device: DevicePlan::new("candle", draft_device.into_string().expect("UTF-8 device"))?,
        },
        None => DraftPlacementPlan::Target,
    };
    let topology = placement.execution_topology(&device);
    let factory = CandleBackendFactory::for_device(&device)?;
    let plan = ExecutionPlan::fully_resident(device).with_drafting(DraftingPlan::External {
        model: draft_path,
        residency: ResidencyPlan::FullyResident,
        placement,
        max_draft_tokens: 3,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let mut model = PreparedTextModel::from_artifact(target, TextModelOptions::default())?
        .select_execution_plan(&factory, &plan, None)?
        .load()?;
    let (target, drafting) = model.parts_mut();
    let eredu_core::RealizedDrafting::External(draft) = drafting else {
        return Err("planned draft was not installed".into());
    };
    let chat = target.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user","content":"Write a short story about a cat who finds a key."})],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let spec: SpeculationOptions = SpeculationOptionsRequest {
        max_draft_tokens: NonZeroUsize::new(3).unwrap(),
        ..Default::default()
    }
    .try_into()?;
    for (do_sample, seed) in [(false, 12), (true, 12), (true, 949)] {
        let settings = PreparedChatGenerationSettings {
            seed,
            overrides: GenerationConfigOverrides {
                do_sample: Some(do_sample),
                temperature: Some(0.85),
                top_k: Some(32),
                top_p: Some(0.91),
                min_p: Some(0.05),
                max_new_tokens: Some(10),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = |settings| GenerationRequest {
            input: PreparedChatInput::RenderedPrompt(&chat),
            output_mode: OutputMode::Text,
            settings,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            control_handle: None,
            on_event: |_: eredu_core::SemanticEvent| {},
        };
        target.reset()?;
        let baseline = target.session().generate(request(settings))?;
        let settings = PreparedChatGenerationSettings {
            decoding_mode: Some(DecodingMode::SelfSpeculation(spec)),
            ..settings
        };
        target.reset()?;
        let ordinary = target
            .drafting_session(SpeculativeDraft::External(draft), Some(spec))
            .generate(request(settings))?;
        let mut captures = 0;
        let charge = std::cell::Cell::new(0);
        let record = |step: eredu::api::ControlledSpeculativeStep| {
            for capture in step.captures {
                assert!(capture.capture.cumulative_usage.captures >= charge.get());
                charge.set(capture.capture.cumulative_usage.captures);
                assert!(capture.capture.records.iter().all(|record| {
                    matches!(
                        record.outcome,
                        CaptureOutcome::Captured { .. } | CaptureOutcome::Skipped { .. }
                    )
                }));
            }
            charge.get()
        };
        target.reset()?;
        let controlled = target
            .drafting_session(SpeculativeDraft::External(draft), Some(spec))
            .with_controlled(
                request(settings),
                GenerationControlOptions {
                    capture: capture_plan(),
                    speculative_snapshots: Some(SnapshotLimits {
                        max_snapshots: 2,
                        max_branches: 1,
                        retained_bytes: 128 << 20,
                        cumulative_copy_bytes: 512 << 20,
                    }),
                    ..Default::default()
                },
                |control| {
                    let session = control.speculative().unwrap();
                    let step = session.step()?.unwrap();
                    assert!(!step.committed_token_ids.is_empty());
                    assert!(!step.captures.is_empty());
                    captures = record(step);
                    assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                    let saved = session.snapshot()?;
                    while let Some(step) = session.step()? {
                        captures = record(step);
                    }
                    let tokens = session.token_ids().to_vec();
                    let charge = session.snapshot_usage().cumulative_copy_bytes;
                    session.restore(&saved)?;
                    while let Some(step) = session.step()? {
                        captures = record(step);
                    }
                    assert_eq!(session.token_ids(), tokens);
                    assert!(session.snapshot_usage().cumulative_copy_bytes > charge);
                    let branch = session.fork(&saved)?;
                    session.exchange(&branch)?;
                    // The branch changes one decision while the parent retains
                    // its independently completed state and sampling history.
                    let next = session.token_ids().len();
                    let token = if tokens[next] == 504 { 2644 } else { 504 };
                    session.force_next_token(token)?;
                    while let Some(step) = session.step()? {
                        captures = record(step);
                    }
                    assert_eq!(session.token_ids()[next], token);
                    session.exchange(&branch)?;
                    assert_eq!(session.token_ids(), tokens);
                    Ok(())
                },
            )?;
        assert_eq!(ordinary.token_ids, controlled.token_ids);
        assert_eq!(ordinary.token_ids.len(), 10);
        if !do_sample {
            assert_eq!(ordinary.token_ids, baseline.token_ids);
        }
        assert!(captures > 0);
        let stats = ordinary
            .stats()
            .drafting
            .as_ref()
            .expect("drafting metrics");
        assert_eq!(stats.execution_topology(), topology);
        assert_eq!(
            controlled.stats().drafting.as_ref().unwrap().execution_topology(),
            topology,
        );
        if !do_sample {
            assert!(stats.accepted_tokens() > 0 && stats.accepted_tokens() < stats.draft_tokens());
        }
        if !do_sample {
            let mut committed = Vec::new();
            target.reset()?;
            let cancelled = target
                .drafting_session(SpeculativeDraft::External(draft), Some(spec))
                .with_controlled(
                    request(settings),
                    GenerationControlOptions::default(),
                    |control| {
                        let session = control.speculative().expect("controlled speculative route");
                        session.step()?.expect("first committed step");
                        committed = session.token_ids().to_vec();
                        session.cancel()?;
                        Ok(())
                    },
                )?;
            assert_eq!(
                cancelled.finish_reason(),
                eredu_core::FinishReason::Cancelled
            );
            assert_eq!(cancelled.token_ids, committed);
        }
        println!(
            "seed={seed} stochastic={do_sample} accepted={}/{} capture_charges={captures} tokens={:?}",
            stats.accepted_tokens(), stats.draft_tokens(), ordinary.token_ids,
        );
    }
    Ok(())
}
