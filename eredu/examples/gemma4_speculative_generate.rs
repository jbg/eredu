//! Compare ordinary and speculative generation with the same prepared chat.
//! Arguments: target directory, assistant directory, prompt, maximum tokens,
//! and the enforced request capacity in bytes.
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};

use anyhow::Context;
use eredu::{
    api::{
        ChatSourceInput, LoadedModel, PreparedChatGenerationSettings, PreparedChatOutputMode,
        PreparedChatPrompt, PreparedChatRequest, PreparedChatSpeculativeGenerationOptions,
        PreparedChatSpeculativeRequest, TokenizerSourceInput, default_local_device,
        local_device_plan,
    },
    runtime::chat::ChatTemplateRequest,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    DraftPlacementPlan, DraftingPlan, ExecutionPlan, GenerationCancellationToken,
    GenerationConfigOverrides, SemanticEvent, TextInferencePolicy,
};

fn main() -> anyhow::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let target_dir = PathBuf::from(args.first().context("target directory required")?);
    let assistant_dir = PathBuf::from(args.get(1).context("assistant directory required")?);
    let prompt = args.get(2).context("prompt required")?;
    let max_tokens = args
        .get(3)
        .context("maximum token count required")?
        .parse::<usize>()?;
    let capacity = args
        .get(4)
        .context("request capacity in bytes required")?
        .parse::<u64>()?;
    anyhow::ensure!(max_tokens > 0, "maximum token count must be positive");

    let plan = ExecutionPlan::fully_resident(local_device_plan(default_local_device())?)
        .with_drafting(DraftingPlan::External {
            model: assistant_dir.display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 3,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let planned =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target_dir, &plan)?;
    let (mut target, mut drafting) = planned.into_parts();
    let cancellation = GenerationCancellationToken::new();
    let tokenizer =
        target.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = target
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancellation,
        )?
        .context("chat preparation cancelled")?;
    let prepared = target
        .prepare_chat(
            &source,
            &ChatTemplateRequest {
                messages: vec![serde_json::json!({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt, "content": prompt}],
                })],
                add_generation_prompt: true,
                ..Default::default()
            },
            capacity,
            &cancellation,
        )?
        .context("chat preparation cancelled")?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(max_tokens),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            managed_memory_capacity_bytes: Some(capacity),
            ..Default::default()
        },
        ..Default::default()
    };
    println!("rendered prompt:\n{}", prepared.rendered_prompt());

    let started = Instant::now();
    let mut text = String::new();
    let ordinary = target
        .start_prepared_chat(PreparedChatRequest::new(&prepared, settings), &cancellation)?
        .context("ordinary generation cancelled")?
        .run(&cancellation, &mut |event| {
            if let SemanticEvent::TextDelta(delta) = event {
                text.push_str(delta.as_str());
            }
        })?;
    println!(
        "ordinary: {} tokens in {:.2?}\n{text}",
        ordinary.token_ids.len(),
        started.elapsed()
    );
    drop(ordinary);

    let mut text = String::new();
    let speculative =
        target.generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
            chat: &prepared,
            input: PreparedChatPrompt::Rendered,
            drafting: drafting
                .as_speculative_draft()
                .context("drafting plan was not realized")?,
            settings,
            output_mode: PreparedChatOutputMode::Semantic,
            skip_special_tokens: true,
            options: PreparedChatSpeculativeGenerationOptions {
                max_draft_tokens: NonZeroUsize::new(3).unwrap(),
                ..Default::default()
            },
            caller_stop_sequences: &[],
            cancellation,
            on_event: |event| {
                if let SemanticEvent::TextDelta(delta) = event {
                    text.push_str(delta.as_str());
                }
            },
        })?;
    println!(
        "speculative: {} tokens in {:.2?}\n{text}",
        speculative.token_ids().len(),
        speculative.stats().elapsed()
    );
    println!(
        "accepted per round: {:?}",
        speculative.stats().accept_lens()
    );
    Ok(())
}
