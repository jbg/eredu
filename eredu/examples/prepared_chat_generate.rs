//! Ordinary, source-funded semantic chat with optional authenticated host media.
//!
//! prepared_chat_generate CHECKPOINT REQUEST.json MEMORY_LIMITS [DEVICE]
//! DEVICE defaults to metal:0. Output is one committed semantic event per line.
//! Caller-owned JSON, processed host arrays and output I/O are outside the
//! framework allocation domain; every framework copy uses the retained source.
#[path = "support/physical_memory.rs"]
mod physical_memory;
use anyhow::{ensure, Context};
use eredu::api::{
    ChatSourceInput, LoadedModel, PreparedChatGenerationSettings, PreparedChatRequest,
    TokenizerSourceInput,
};
use eredu::runtime::chat::{ChatTemplateRequest, ToolChoice};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    DevicePlan, ExecutionPlan, GenerationCancellationToken, GenerationConfigOverrides, InputExtent,
    InputMetadataKey, InputModality, InputPayloadKind, TextInferencePolicy,
};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView};
use serde::Deserialize;
use serde_json::Value;
use std::{fs::File, io::Write, num::NonZeroU64, path::PathBuf};

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Choice {
    None,
    Auto,
    Required,
}
#[derive(Deserialize)]
struct Request {
    messages: Vec<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Choice,
    max_tokens: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    prefill_chunk_positions: Option<NonZeroU64>,
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    /// Model-appropriate processor output, including ordered text slots.
    /// The architecture validates the supplied shapes and semantic coordinates.
    #[serde(default)]
    prepared_parts: Vec<Part>,
}
#[derive(Deserialize)]
struct Part {
    modality: InputModality,
    kind: InputPayloadKind,
    payload: Tensor,
    #[serde(default)]
    metadata: Vec<(InputMetadataKey, Tensor)>,
    #[serde(default)]
    extents: Vec<InputExtent>,
}
#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    values: Values,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Values {
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
}
impl Tensor {
    fn view(&self) -> HostTensorView<'_> {
        HostTensorView {
            shape: &self.shape,
            values: match &self.values {
                Values::U32(values) => HostTensorValues::U32(values),
                Values::I32(values) => HostTensorValues::I32(values),
                Values::F32(values) => HostTensorValues::F32(values),
                Values::Bool(values) => HostTensorValues::Bool(values),
            },
        }
    }
}
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "prepared_chat_generate CHECKPOINT REQUEST.json MEMORY_LIMITS [DEVICE]";
    let checkpoint = PathBuf::from(args.next().context(usage)?);
    let input = File::open(args.next().context(usage)?)?;
    let capacity = physical_memory::parse(&args.next().context(usage)?)?;
    let device = args.next().unwrap_or_else(|| "metal:0".into());
    ensure!(args.next().is_none(), "{usage}");
    let request: Request = serde_json::from_reader(input)?;
    ensure!(request.max_tokens > 0, "max_tokens must be positive");
    let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", device)?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &checkpoint, &plan)?
            .into_parts();
    let cancellation = GenerationCancellationToken::new();
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            !request.tools.is_empty(),
            &cancellation,
        )?
        .context("cancelled before template compilation")?;
    let policy = ChatTemplateRequest {
        messages: request.messages,
        tools: request.tools,
        tool_choice: match request.tool_choice {
            Choice::None => ToolChoice::None,
            Choice::Auto => ToolChoice::Auto,
            Choice::Required => ToolChoice::Required,
        },
        enable_thinking: request.enable_thinking,
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, &capacity, &cancellation)?
        .context("cancelled before chat preparation")?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(request.max_tokens),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            memory_limits: capacity.clone(),
            prefill_chunk_positions: request.prefill_chunk_positions,
            ..Default::default()
        },
        seed: request.seed,
        ..Default::default()
    };
    let mut generation = PreparedChatRequest::new(&chat, settings.clone());
    generation.stop_sequences = &request.stop_sequences;
    if !request.prepared_parts.is_empty() {
        // These small borrowing views remain application-owned. The public
        // preparation method funds and authenticates all framework copies once.
        let metadata: Vec<Vec<_>> = request
            .prepared_parts
            .iter()
            .map(|part| {
                part.metadata
                    .iter()
                    .map(|(key, tensor)| (*key, tensor.view()))
                    .collect()
            })
            .collect();
        let parts: Vec<_> = request
            .prepared_parts
            .iter()
            .zip(&metadata)
            .map(|(part, metadata)| HostInputPart {
                modality: part.modality,
                kind: part.kind,
                payload: part.payload.view(),
                metadata,
                extents: &part.extents,
            })
            .collect();
        generation.input = eredu::api::PreparedChatPrompt::Media(
            model
                .prepare_chat_input(&chat, &parts, &cancellation)?
                .context("cancelled before media preparation")?,
        );
    }
    let session = model
        .start_prepared_chat(generation, &cancellation)?
        .context("cancelled before generation")?;
    let mut output = std::io::stdout().lock();
    let mut output_failure = None;
    let mut emit = |event| {
        if output_failure.is_some() {
            return;
        }
        if let Err(error) = serde_json::to_writer(&mut output, &event)
            .map_err(std::io::Error::other)
            .and_then(|()| output.write_all(b"\n"))
        {
            output_failure = Some(error);
            cancellation.cancel();
        }
    };
    let result = session.run(&cancellation, &mut emit)?;
    if let Some(error) = output_failure {
        return Err(error.into());
    }
    output.flush()?;
    eprintln!(
        "committed_tokens={} finish={:?}",
        result.token_ids.len(),
        result.finish_reason
    );
    Ok(())
}
