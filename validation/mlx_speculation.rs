//! Released SmolLM2 greedy speculation acceptance through the MLX facade.

use eredu::api::{
    GenerationControlOptions, GenerationRequest, LoadedModel, OutputMode,
    PreparedChatGenerationSettings, PreparedChatInput, SpeculativeProposalDisposition,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    decoding::{SpeculationOptions, SpeculationOptionsRequest},
    DevicePlan, DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationConfigOverrides,
    ResidencyPlan,
};
use std::{num::NonZeroUsize, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let target_path = PathBuf::from(args.next().expect("BF16 SafeTensors target directory"));
    let draft_path = args
        .next()
        .expect("Q4_K_M draft file")
        .into_string()
        .expect("draft path is UTF-8");
    let device = args
        .next()
        .map(|value| value.into_string().expect("device is UTF-8"))
        .unwrap_or_else(|| "metal:0".into());
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", &device)?).with_drafting(
        DraftingPlan::External {
            model: draft_path,
            residency: ResidencyPlan::FullyResident,
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 3,
            lookahead: false,
            adaptive_lookahead: false,
        },
    );
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target_path, &plan, None)?;
    let (target, drafting) = loaded.parts_mut();
    let chat = target.prepare_chat(ChatTemplateRequest {
        messages: vec![serde_json::json!({
            "role": "user",
            "content": "Write a short story about a cat who finds a key."
        })],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let prompt_ids = target.encode(chat.rendered_prompt(), false)?;
    let settings = PreparedChatGenerationSettings {
        seed: 12,
        overrides: GenerationConfigOverrides {
            do_sample: Some(false),
            repetition_penalty: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            max_new_tokens: Some(10),
            ..Default::default()
        },
        ..Default::default()
    };
    let request = || GenerationRequest {
        input: PreparedChatInput::token_ids(&chat, prompt_ids.clone()),
        output_mode: OutputMode::Text,
        settings,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        control_handle: None,
        on_event: |_: eredu_core::SemanticEvent| {},
    };
    let options: SpeculationOptions = SpeculationOptionsRequest {
        max_draft_tokens: NonZeroUsize::new(3).unwrap(),
        ..Default::default()
    }
    .try_into()?;

    // Speculation forks the installed target prefix. Give all three runs
    // identical state and input rather than continuing the previous run.
    target.reset()?;
    let baseline = target.session().generate(request())?;
    target.reset()?;
    let ordinary = target
        .drafting_session(drafting.as_speculative_draft().unwrap(), Some(options))
        .generate(request())?;
    target.reset()?;
    let mut rejected = false;
    let mut committed = Vec::new();
    let controlled = target
        .drafting_session(drafting.as_speculative_draft().unwrap(), Some(options))
        .with_controlled(request(), GenerationControlOptions::default(), |control| {
            let session = control.speculative().unwrap();
            while let Some(step) = session.step()? {
                if let Some(verification) = step.verification {
                    rejected |= verification
                        .dispositions
                        .contains(&SpeculativeProposalDisposition::Rejected);
                }
                committed.extend(step.committed_token_ids);
            }
            Ok(())
        })?;

    let expected = [6403, 1980, 253, 655, 28, 281, 253, 18840, 1838, 2291];
    assert_eq!(baseline.token_ids, expected);
    assert_eq!(ordinary.token_ids, baseline.token_ids);
    assert_eq!(controlled.token_ids, baseline.token_ids);
    assert_eq!(committed, baseline.token_ids);
    assert!(rejected);
    let stats = ordinary.stats().drafting.as_ref().unwrap();
    assert!(stats.accepted_tokens() > 0 && stats.accepted_tokens() < stats.draft_tokens());
    println!(
        "{device} tokens={:?} accepted={}/{}",
        controlled.token_ids,
        stats.accepted_tokens(),
        stats.draft_tokens()
    );
    Ok(())
}
