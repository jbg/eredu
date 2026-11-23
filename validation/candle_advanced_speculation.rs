//! Released Candle acceptance for embedded and conditioned drafting.

use eredu::api::{
    inspect_artifact, ArtifactSource, GenerationControlOptions, GenerationRequest, LoadedModel,
    OutputMode, PreparedChatGenerationSettings, PreparedChatInput, PreparedTextModel,
    SpeculativeActivationPhase, SpeculativeActivationPlan, TextModelOptions,
    SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_candle::{CandleBackendFactory, CandleContext};
use eredu_core::{
    capture::{
        CaptureInvocationBounds, CaptureLimitPolicy, CaptureLimits, CapturePlan, CaptureProvenance,
        CaptureSelection, CaptureTransform, CaptureUsage, CAPTURE_SCHEMA_VERSION,
    },
    decoding::{SpeculationOptions, SpeculationOptionsRequest},
    execution_control::SnapshotLimits,
    intervention::{
        InterventionAction, InterventionDtype, InterventionEvidence, InterventionOperation,
        InterventionOutcome, InterventionPlan, INTERVENTION_SCHEMA_VERSION,
    },
    DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationConfigOverrides,
    ModelLoadOptions, RealizedDrafting, ResidencyPlan,
};
use eredu_runtime::ModelLoadFacts;
use eredu_runtime::{
    speculative::autoregressive::{LocalCausalDraft, LocalCausalSession},
    NormalizedLoadRequest,
};

type Target = LoadedModel<CandleContext, LocalCausalSession<CandleContext>>;
type Drafting = RealizedDrafting<LocalCausalDraft<CandleContext>>;
type ProbeResult<T> = Result<T, Box<dyn std::error::Error>>;

const TARGET_POINT: &str = "model.layers.3.attention.write";
const PREDICTION_POINT: &str = "mtp.layers.0.prediction.readout.normalized";

fn activation_plan() -> SpeculativeActivationPlan {
    let limit = CaptureUsage {
        captures: 24,
        retained_bytes: 256 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 1 << 20,
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            provenance: CaptureProvenance::ContentExact,
            selections: [TARGET_POINT, PREDICTION_POINT]
                .into_iter()
                .map(|path| CaptureSelection {
                    id: path.into(),
                    path: path.into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::Preview { max_elements: 8 },
                })
                .collect(),
            limits: CaptureLimits {
                per_step: limit,
                cumulative: limit.checked_mul(24).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations: vec![InterventionOperation {
                id: "scale-prediction".into(),
                target: PREDICTION_POINT.into(),
                schedule: Default::default(),
                slices: vec![],
                action: InterventionAction::Scale {
                    dtype: InterventionDtype::Bfloat16,
                    factor: 0.5,
                },
                evidence: InterventionEvidence::Preview { max_elements: 8 },
            }],
        },
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 64,
            max_context: None,
            max_predictions: 6,
        },
    }
}

fn exercise(target: &mut Target, drafting: &mut Drafting, depth: usize) -> ProbeResult<()> {
    use SpeculativeActivationPhase as Phase;

    let chat = target.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({
            "role":"user",
            "content":"What is the capital of France?"
        })],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let embedded = !drafting.is_external();
    let options: SpeculationOptions = SpeculationOptionsRequest {
        max_draft_tokens: depth.try_into().unwrap(),
        ..Default::default()
    }
    .try_into()?;
    for do_sample in [false, true] {
        let settings = PreparedChatGenerationSettings {
            seed: 12,
            overrides: GenerationConfigOverrides {
                do_sample: Some(do_sample),
                max_new_tokens: Some(if embedded { 5 } else { 6 }),
                temperature: Some(0.85),
                top_k: Some(32),
                top_p: Some(0.95),
                min_p: Some(0.05),
                repetition_penalty: Some(1.0),
                repeat_last_n: Some(64),
                frequency_penalty: Some(0.0),
                presence_penalty: Some(0.0),
            },
            ..Default::default()
        };
        let request = || GenerationRequest {
            input: PreparedChatInput::RenderedPrompt(&chat),
            output_mode: OutputMode::Text,
            settings,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            control_handle: None,
            on_event: |_: eredu_core::SemanticEvent| {},
        };
        target.reset()?;
        let baseline = target.session().generate(request())?;
        target.reset()?;
        let ordinary = target
            .drafting_session(drafting.as_speculative_draft().unwrap(), Some(options))
            .generate(request())?;
        let instrumented = embedded && !do_sample;
        let activations = instrumented
            .then(|| target.prepare_speculative_activations(activation_plan()))
            .transpose()?;
        target.reset()?;

        let mut applied = 0;
        let mut proposals = 0;
        let controlled = target
            .drafting_session(drafting.as_speculative_draft().unwrap(), Some(options))
            .with_controlled(
                request(),
                GenerationControlOptions {
                    speculative_activations: activations,
                    speculative_snapshots: Some(SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 2 << 30,
                        cumulative_copy_bytes: 4 << 30,
                    }),
                    ..Default::default()
                },
                |control| {
                    let session = control.speculative().unwrap();
                    let initial = session.step()?.unwrap();
                    if instrumented {
                        assert!(initial.activations.iter().any(|activation| {
                            activation.phase == Phase::TargetPrefill
                                && activation.captures.records.iter().any(|record| {
                                    record.path == TARGET_POINT && record.payload.is_some()
                                })
                        }));
                    }
                    let saved = session.snapshot()?;
                    let branch = instrumented.then(|| session.fork(&saved)).transpose()?;
                    let branch_position = session.token_ids().len();
                    let mut last_invocation = 0;
                    let mut encoded = 0;
                    let mut finish =
                        |session: &mut dyn eredu::api::ControlledSpeculativeSession| {
                            while let Some(step) = session.step()? {
                                for activation in step.activations {
                                    assert!(activation.invocation > last_invocation);
                                    last_invocation = activation.invocation;
                                    assert!(
                                        activation.captures.cumulative_usage.encoded_bytes
                                            >= encoded
                                    );
                                    encoded = activation.captures.cumulative_usage.encoded_bytes;
                                    if activation.phase == (Phase::Proposal { depth: 0 }) {
                                        proposals += 1;
                                    }
                                    for edit in activation.captures.interventions {
                                        if edit.outcome == InterventionOutcome::Applied {
                                            assert!(matches!(
                                                activation.phase,
                                                Phase::Proposal { depth: 0 }
                                                    | Phase::PredictionPrefill
                                                    | Phase::PredictionReplay
                                            ));
                                            assert_eq!(edit.evidence.len(), 2);
                                            assert_ne!(
                                                edit.evidence[0].payload,
                                                edit.evidence[1].payload
                                            );
                                            applied += 1;
                                        }
                                    }
                                }
                            }
                            Ok::<_, eredu::api::SpeculativeControlError>(())
                        };
                    finish(session)?;
                    let tokens = session.token_ids().to_vec();
                    let copy_bytes = session.snapshot_usage().cumulative_copy_bytes;
                    session.restore(&saved)?;
                    assert!(session.snapshot_usage().cumulative_copy_bytes > copy_bytes);
                    finish(session)?;
                    assert_eq!(session.token_ids(), tokens);
                    if let Some(branch) = branch {
                        session.exchange(&branch)?;
                        let forced = if tokens[branch_position] == 42 {
                            43
                        } else {
                            42
                        };
                        session.force_next_token(forced)?;
                        finish(session)?;
                        assert_eq!(session.token_ids()[branch_position], forced);
                        session.exchange(&branch)?;
                        assert_eq!(session.token_ids(), tokens);
                    }
                    Ok(())
                },
            )?;
        assert_eq!(controlled.token_ids, ordinary.token_ids);
        let stats = ordinary.stats().drafting.as_ref().unwrap();
        assert!(stats.draft_tokens() > 0);
        if !do_sample {
            assert_eq!(baseline.token_ids, ordinary.token_ids);
            assert!(stats.accepted_tokens() > 0);
        }
        if instrumented {
            assert!(proposals > 0 && applied > 0);
        }
        println!(
            "embedded={embedded} stochastic={do_sample} accepted={}/{} edits={applied} tokens={:?}",
            stats.accepted_tokens(),
            stats.draft_tokens(),
            ordinary.token_ids,
        );
    }
    Ok(())
}

fn run() -> ProbeResult<()> {
    let mut arguments = std::env::args_os().skip(1);
    let embedded_path = arguments.next().expect("embedded checkpoint directory");
    let target_path = arguments.next().expect("conditioned target GGUF file");
    let assistant_path = arguments.next().expect("conditioned assistant directory");
    let placement = arguments.next().unwrap_or_else(|| "cpu:0".into());
    let placement = placement.into_string().expect("device placement is UTF-8");
    let device = DevicePlan::new("candle", placement)?;
    let placement = match arguments.next() {
        Some(draft) => DraftPlacementPlan::Device {
            device: DevicePlan::new("candle", draft.into_string().expect("UTF-8 device"))?,
        },
        None => DraftPlacementPlan::Target,
    };

    if embedded_path != "-" {
        let embedded = inspect_artifact(&ArtifactSource::new(embedded_path))?;
        let draft_plan = DraftingPlan::Embedded {
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        };
        let request = NormalizedLoadRequest::default().with_drafting_plan(&draft_plan)?;
        let mut target = PreparedTextModel::from_artifact(embedded, TextModelOptions::default())?
            .load_on::<_, ModelLoadFacts<_>>(
            CandleContext::open(&device)?,
            ModelLoadOptions::new(request),
        )?;
        exercise(&mut target, &mut RealizedDrafting::Embedded, 1)?;
        drop(target);
    }

    let target = inspect_artifact(&ArtifactSource::new(target_path))?;
    let plan = ExecutionPlan::fully_resident(device).with_drafting(DraftingPlan::External {
        model: assistant_path
            .into_string()
            .expect("assistant path is UTF-8"),
        residency: ResidencyPlan::FullyResident,
        placement,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let factory = CandleBackendFactory::for_device(plan.device())?;
    let mut model = PreparedTextModel::from_artifact(target, TextModelOptions::default())?
        .select_execution_plan(&factory, &plan, None)?
        .load()?;
    let (target, draft) = model.parts_mut();
    exercise(target, draft, 2)
}

fn main() {
    std::thread::Builder::new()
        .stack_size(96 << 20)
        .spawn(|| run().unwrap())
        .unwrap()
        .join()
        .unwrap();
}
