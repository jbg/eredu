use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use eredu::{
    api::{
        default_local_device, local_device_plan, ChatTemplateRequest, LocalBackendFactory,
        LocalModel, LocalPreparedChatInput, LocalPreparedChatMtpGenerationRequest, PreparedChat,
        PreparedChatGenerationSettings, PreparedChatMtpGenerationOptions,
    },
    DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationCancellationToken,
    GenerationConfigOverrides, TextGenerationConfig,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let target_dir = args
        .first()
        .map(PathBuf::from)
        .or_else(default_target_snapshot)
        .expect("target model dir required");
    let assistant_dir = args
        .get(1)
        .map(PathBuf::from)
        .or_else(default_assistant_snapshot)
        .expect("assistant model dir required");
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Why is the sky blue?".to_string());
    let max_tokens = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(96);

    println!("target: {}", target_dir.display());
    println!("assistant: {}", assistant_dir.display());
    println!("prompt: {prompt:?}");

    let prepared = prepare_prompt(&target_dir, &prompt)?;
    println!(
        "\n=== rendered prompt ===\n{}\n",
        prepared.rendered_prompt()
    );

    let greedy = run_greedy(&target_dir, prepared.rendered_prompt(), max_tokens)?;
    println!("\n=== greedy ===");
    println!(
        "tokens: {} elapsed: {:.2?}",
        greedy.token_ids.len(),
        greedy.elapsed
    );
    println!("{}", greedy.text);

    let mtp = run_mtp(&target_dir, &assistant_dir, &prepared, max_tokens)?;
    println!("\n=== mtp ===");
    println!(
        "tokens: {} elapsed: {:.2?}",
        mtp.token_ids.len(),
        mtp.elapsed
    );
    println!("accepted per round: {:?}", mtp.accept_lens);
    println!("{}", mtp.text);

    Ok(())
}

struct GenerationResult {
    token_ids: Vec<u32>,
    text: String,
    elapsed: std::time::Duration,
    accept_lens: Vec<usize>,
}

fn prepare_prompt(target_dir: &PathBuf, prompt: &str) -> anyhow::Result<PreparedChat> {
    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), target_dir, &plan)?;
    let (mut loaded, _) = planned.into_parts();
    loaded
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": prompt, "content": prompt}],
            })],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        })
        .map_err(Into::into)
}

fn run_greedy(
    target_dir: &PathBuf,
    prompt: &str,
    max_tokens: usize,
) -> anyhow::Result<GenerationResult> {
    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    let planned =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), target_dir, &plan)?;
    let (mut loaded, _) = planned.into_parts();
    let prompt_tokens = loaded.encode(prompt, false)?;
    let eos = loaded.eos_token_ids().to_vec();
    let mut ids = Vec::new();
    let start = Instant::now();
    {
        let resolved = loaded.resolve_generation_config(GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(max_tokens),
            ..Default::default()
        })?;
        let generator = loaded
            .generate_tokens(prompt_tokens, TextGenerationConfig::new(resolved))?
            .take(max_tokens);
        for token in generator {
            let id = token?;
            if eos.contains(&id) {
                break;
            }
            ids.push(id);
        }
    }
    let elapsed = start.elapsed();
    let text = loaded.decode(&ids, true)?;
    Ok(GenerationResult {
        token_ids: ids,
        text,
        elapsed,
        accept_lens: Vec::new(),
    })
}

fn run_mtp(
    target_dir: &Path,
    assistant_dir: &Path,
    prepared: &PreparedChat,
    max_tokens: usize,
) -> anyhow::Result<GenerationResult> {
    let mut plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?);
    plan.drafting = DraftingPlan::External {
        model: assistant_dir.display().to_string(),
        placement: DraftPlacementPlan::Target,
        max_draft_tokens: 3,
        lookahead: false,
        adaptive_lookahead: false,
    };
    let planned =
        LocalModel::load_execution_plan(&LocalBackendFactory::default(), target_dir, &plan)?;
    let (mut target, mut drafting) = planned.into_parts();
    if !drafting.is_enabled() {
        anyhow::bail!("external drafting plan was not realized");
    }
    let output = target.generate_prepared_chat_mtp(LocalPreparedChatMtpGenerationRequest {
        input: LocalPreparedChatInput::rendered_prompt(prepared),
        drafting: &mut drafting,
        settings: PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(max_tokens),
                ..Default::default()
            },
            ..PreparedChatGenerationSettings::default()
        },
        options: PreparedChatMtpGenerationOptions {
            max_draft_tokens: NonZeroUsize::new(3).unwrap(),
            ..PreparedChatMtpGenerationOptions::default()
        },
        caller_stop_sequences: &[],
        cancellation: GenerationCancellationToken::new(),
        on_event: |_| {},
    })?;
    let mut generated = output.token_ids;
    let stats = output.stats;
    if generated
        .last()
        .is_some_and(|token| target.eos_token_ids().contains(token))
    {
        generated.pop();
    }
    let text = target.decode(&generated, true)?;
    Ok(GenerationResult {
        token_ids: generated,
        text,
        elapsed: stats.elapsed,
        accept_lens: stats.accept_lens,
    })
}

fn default_target_snapshot() -> Option<PathBuf> {
    default_snapshot("models--mlx-community--gemma-4-e4b-it-4bit")
}

fn default_assistant_snapshot() -> Option<PathBuf> {
    default_snapshot("models--mlx-community--gemma-4-e4b-it-assistant-bf16")
}

fn default_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let snapshots = PathBuf::from(std::env::var_os("HOME")?)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    snapshots
        .read_dir()
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("config.json").exists())
}
