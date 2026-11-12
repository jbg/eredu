//! File decoding and typed visual prompts for ordinary CLI generation.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use eredu::api::{
    ChatInputRequest, ChatMessage, ChatPart, ChatPreparationOptions, ChatToolDefinition,
    LoadedModel, PreparedChatRequest,
};
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_core::{
    ModelCapabilityBackend, MultimodalPreparationBackend, RgbImage, Video, VideoSampling,
};
use image::{ImageFormat, ImageReader};

#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct MediaArgs {
    /// Append a JPEG or PNG image to the user prompt. May be repeated.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["raw", "decisions", "expert_cache_benchmark", "estimate_memory_tokens"])]
    image: Vec<PathBuf>,

    /// Append a clip from a directory of JPEG or PNG frames in filename order.
    #[arg(long, value_name = "DIRECTORY", requires = "video_fps", conflicts_with_all = ["raw", "decisions", "expert_cache_benchmark", "estimate_memory_tokens"])]
    video_frames: Option<PathBuf>,

    /// Source frame rate of the image sequence supplied with --video-frames.
    #[arg(long, value_name = "FPS", requires = "video_frames")]
    video_fps: Option<f64>,
}

impl MediaArgs {
    fn is_empty(&self) -> bool {
        self.image.is_empty() && self.video_frames.is_none()
    }

    pub(crate) fn validate(&self, auto: Option<super::AutoMode>) -> Result<()> {
        if self
            .video_fps
            .is_some_and(|fps| !fps.is_finite() || fps <= 0.0)
        {
            bail!("--video-fps must be finite and greater than zero");
        }
        if !self.is_empty() && auto == Some(super::AutoMode::Benchmark) {
            bail!("--auto benchmark measures text-only prompts; use --auto quick or --no-auto with visual input");
        }
        Ok(())
    }

    pub(crate) fn prepare<B: MultimodalPreparationBackend + ModelCapabilityBackend>(
        &self,
        model: &mut LoadedModel<B>,
        text: &ChatTemplateRequest,
        prompt: &str,
    ) -> Result<Option<PreparedChatRequest<B>>> {
        if self.is_empty() {
            return Ok(None);
        }
        let mut parts = vec![ChatPart::Text(prompt.to_owned())];
        for image in &self.image {
            parts.push(ChatPart::Image(decode_image(image)?));
        }
        if let Some(directory) = &self.video_frames {
            parts.push(ChatPart::Video(decode_video(directory, self.video_fps)?));
        }
        let tools = text
            .tools
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                let Tool::Function { function } = serde_json::from_value(tool.clone())
                    .with_context(|| {
                        format!("invalid OpenAI-shaped tool definition at index {index}")
                    })?;
                Ok(ChatToolDefinition {
                    name: function.name,
                    description: function.description,
                    parameters: function.parameters,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let request = ChatInputRequest {
            messages: vec![ChatMessage::User(parts)],
            tools,
            options: ChatPreparationOptions {
                tool_choice: text.tool_choice.clone(),
                parallel_tool_calls: text.parallel_tool_calls.clone(),
                enable_thinking: text.enable_thinking,
                reasoning_effort: text.reasoning_effort.clone(),
                allow_unparsed_reasoning: text.allow_unparsed_reasoning,
                add_generation_prompt: text.add_generation_prompt,
                ..Default::default()
            },
        };
        Ok(Some(model.prepare_chat_input(request)?))
    }
}

#[derive(serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Tool {
    Function { function: Function },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Function {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

fn decode_image(path: &Path) -> Result<RgbImage> {
    let image = ImageReader::open(path)
        .with_context(|| format!("failed to open image {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to identify image {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode image {}", path.display()))?
        .into_rgb8();
    let (width, height) = image.dimensions();
    RgbImage::new(image.into_raw(), width, height).map_err(Into::into)
}

fn decode_video(directory: &Path, fps: Option<f64>) -> Result<Video> {
    let entries = std::fs::read_dir(directory)
        .with_context(|| format!("failed to list video frames in {}", directory.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.is_file()
            && matches!(
                ImageFormat::from_path(&path),
                Ok(ImageFormat::Jpeg | ImageFormat::Png)
            )
        {
            paths.push(path);
        }
    }
    paths.sort();
    let frames = paths
        .iter()
        .map(|path| decode_image(path))
        .collect::<Result<Vec<_>>>()?;
    Video::new(frames, fps, VideoSampling::ProcessorDefault)
        .with_context(|| format!("invalid video frames in {}", directory.display()))
}
