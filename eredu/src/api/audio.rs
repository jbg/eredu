//! Turn-based audio tasks over selected realtime execution.

use std::{error::Error, sync::Arc};

use eredu_text::tokenizer::{ModelChatTemplate, Tokenizer};
use serde_json::json;

use super::{
    realtime::{
        prepare_mlx_realtime, AudioAdvance, AudioIngressRequest, AudioMode, AudioRequest,
        LoadedMlxRealtime, MlxAudioIngress, MlxAudioModel, MlxAudioPcmTranscription,
        MlxAudioSession, PreparedMlxRealtime, PreparedRealtimeArtifact, RealtimeLoadOptions,
    },
    TextDecoder,
};
use eredu_backend_mlx::native::ExecutionContext;

type TaskResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Task for one audio request.
pub enum AudioTask<'a> {
    /// Transcribe incoming audio as it arrives.
    Transcribe,
    /// Answer a spoken request. The optional instruction accompanies the audio.
    Respond {
        /// Optional instruction paired with the utterance.
        instruction: &'a str,
    },
    /// Synthesize speech from text.
    Speak {
        /// Text to synthesize.
        text: &'a str,
    },
    /// Translate incoming speech into text in the specified language.
    Translate {
        /// Language spoken in the input.
        source_language: &'a str,
        /// Target language for the translated text.
        language: &'a str,
    },
}

/// Published output from a turn-based task.
pub enum AudioTaskOutput {
    /// An incremental text fragment.
    Text(String),
    /// Newly available mono PCM.
    Pcm(Vec<f32>),
}

enum Prompt {
    Transcribe,
    Text(Vec<i32>),
    Audio(Vec<i32>, Vec<i32>, AudioMode),
}

/// Cold task selection and prompt preparation.
pub struct PreparedAudioTask<'a> {
    selected: PreparedMlxRealtime<'a>,
    prompt: Prompt,
    decoder: TextDecoder,
    input_rate: Option<u32>,
    output_rate: Option<u32>,
    max_slots: usize,
    seed: u64,
    speech_only: bool,
}

impl<'a> PreparedAudioTask<'a> {
    /// Resolves a task using the retained model assets and selected backend.
    pub fn new(
        artifact: &'a PreparedRealtimeArtifact,
        task: AudioTask<'_>,
        max_slots: usize,
        seed: u64,
    ) -> TaskResult<Self> {
        let PreparedRealtimeArtifact::Lfm3Audio(audio) = artifact else {
            return Err("the selected model has no turn-based audio adapter".into());
        };
        if max_slots == 0 {
            return Err("audio response budget must be positive".into());
        }
        let assets = &audio.text_assets;
        let mut tokenizer = Tokenizer::from_bytes(&assets.tokenizer)?;
        let speech_only = matches!(task, AudioTask::Speak { .. });
        let input_rate = (!speech_only).then_some(audio.processor.sample_rate());
        let output_rate = matches!(task, AudioTask::Speak { .. } | AudioTask::Respond { .. })
            .then_some(u32::try_from(audio.codec.config().sampling_rate)?);
        let (content, system, mode) = match task {
            AudioTask::Transcribe => (None, None, AudioMode::Text),
            AudioTask::Respond { instruction } => (
                Some(json!([{"type":"audio"}, {"type":"text","text":instruction}])),
                None,
                AudioMode::Speech,
            ),
            AudioTask::Translate { source_language, language } => (
                Some(json!([{"type":"audio"}])),
                Some(format!(
                    "Translate the following speech from {source_language} to {language} in text form."
                )),
                AudioMode::Text,
            ),
            AudioTask::Speak { text } => (
                Some(json!(text)),
                Some("Perform TTS in American English using a female voice.".into()),
                AudioMode::Speech,
            ),
        };
        let prompt = if let Some(content) = content {
            let mut messages = Vec::new();
            let named_tts = speech_only
                .then(|| assets.named_chat_templates.get("tts"))
                .flatten();
            if let Some(system) = system.filter(|_| named_tts.is_none()) {
                messages.push(json!({"role":"system","content":system}));
            }
            messages.push(json!({"role":"user","content":content}));
            let mut kwargs: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&assets.special_tokens)?;
            kwargs.retain(|key, _| key.ends_with("_token"));
            let rendered = tokenizer
                .apply_chat_template_json(
                    ModelChatTemplate::Single(
                        named_tts.unwrap_or(&assets.chat_template).to_string(),
                    ),
                    [messages],
                    None,
                    "turn-based-audio",
                    true,
                    Some(&kwargs),
                )?
                .pop()
                .ok_or("chat template returned no prompt")?;
            let ids = tokenizer
                .encode(rendered, false)?
                .get_ids()
                .iter()
                .copied()
                .map(i32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            if ids.contains(&audio.tokens.output) {
                return Err("the prompt contains a reserved speech output token".into());
            }
            let mut spans = ids.split(|id| *id == audio.tokens.input);
            let prefix = spans.next().expect("split yields at least one span");
            match (input_rate, spans.next(), spans.next()) {
                (None, None, None) => Prompt::Text(ids),
                (Some(_), Some(suffix), None) => {
                    Prompt::Audio(prefix.to_vec(), suffix.to_vec(), mode)
                }
                _ => {
                    return Err(
                        "the chat template must render exactly one placeholder per audio input"
                            .into(),
                    )
                }
            }
        } else {
            Prompt::Transcribe
        };
        let decoder = TextDecoder::from_shared_tokenizer(Arc::new((*tokenizer).clone()), true);
        Ok(Self {
            selected: prepare_mlx_realtime(artifact, &RealtimeLoadOptions::Lfm3Audio)?,
            prompt,
            decoder,
            input_rate,
            output_rate,
            max_slots,
            seed,
            speech_only,
        })
    }

    /// Required mono input sample rate, when the task accepts PCM.
    pub fn input_sample_rate(&self) -> Option<u32> {
        self.input_rate
    }

    /// Mono output sample rate, when the task produces PCM.
    pub fn output_sample_rate(&self) -> Option<u32> {
        self.output_rate
    }

    /// Binds the selected model and starts the task.
    pub fn load(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> TaskResult<AudioTaskSession> {
        let LoadedMlxRealtime::Lfm3Audio(model) = self.selected.load(weights, execution)? else {
            return Err("loaded audio task differs from its selection".into());
        };
        let state = match self.prompt {
            Prompt::Transcribe => State::Transcribe(model.prepare_pcm_transcription()?),
            Prompt::Text(tokens) => {
                State::Response(start_text(*model, &tokens, self.max_slots, self.seed)?)
            }
            Prompt::Audio(prefix, suffix, mode) => State::Ingress(model.prepare_ingress(
                AudioIngressRequest {
                    prefix: &prefix,
                    suffix: &suffix,
                    mode,
                    max_slots: self.max_slots,
                },
                self.seed,
            )?),
        };
        Ok(AudioTaskSession {
            state,
            decoder: self.decoder,
            input_rate: self.input_rate,
            pending_text: None,
            speech_only: self.speech_only,
        })
    }
}

fn start_text(
    model: MlxAudioModel,
    tokens: &[i32],
    max_slots: usize,
    seed: u64,
) -> TaskResult<MlxAudioSession> {
    Ok(model.start(
        AudioRequest {
            tokens,
            features: &[],
            replay: &[],
            mode: AudioMode::Speech,
            max_slots,
        },
        seed,
    )?)
}

enum State {
    Transcribe(MlxAudioPcmTranscription),
    Ingress(MlxAudioIngress),
    Response(MlxAudioSession),
    Finished,
}

/// One loaded audio request and its incremental output decoder.
pub struct AudioTaskSession {
    state: State,
    decoder: TextDecoder,
    input_rate: Option<u32>,
    pending_text: Option<String>,
    speech_only: bool,
}

impl AudioTaskSession {
    /// Submits mono PCM at the selected rate and returns any ready transcript.
    pub fn push(&mut self, mut samples: &[f32]) -> TaskResult<String> {
        let rate = self.input_rate.ok_or("this task takes no PCM input")?;
        let mut text = String::new();
        match &mut self.state {
            State::Ingress(ingress) => {
                ingress.submit(samples, rate)?;
            }
            State::Transcribe(session) => {
                while !samples.is_empty() {
                    let count = samples.len().min(session.available_samples());
                    if count != 0 {
                        session.submit(&samples[..count], rate)?;
                        samples = &samples[count..];
                    }
                    let block = session.advance(false)?;
                    if count == 0 && block.is_none() {
                        return Err("transcription cannot accept more PCM".into());
                    }
                    if let Some(block) = block {
                        append_text(&mut self.decoder, block.tokens, &mut text)?;
                    }
                }
            }
            _ => return Err("audio input is closed".into()),
        }
        Ok(text)
    }

    /// Closes a PCM utterance and prepares the final transcript or response.
    pub fn finish_input(&mut self) -> TaskResult<()> {
        match &mut self.state {
            State::Transcribe(session) => session.finish_input()?,
            State::Ingress(_) => {
                let State::Ingress(ingress) = std::mem::replace(&mut self.state, State::Finished)
                else {
                    unreachable!()
                };
                self.state = State::Response(ingress.finish()?);
            }
            _ => return Err("this task has no open PCM input".into()),
        }
        Ok(())
    }

    /// Returns the next output after input closure.
    pub fn next_output(&mut self) -> TaskResult<Option<AudioTaskOutput>> {
        if let Some(text) = self.pending_text.take() {
            return Ok(Some(AudioTaskOutput::Text(text)));
        }
        loop {
            match &mut self.state {
                State::Transcribe(session) => {
                    if session.is_finished() {
                        self.state = State::Finished;
                        continue;
                    }
                    let Some(block) = session.advance(false)? else {
                        if session.is_finished() {
                            self.state = State::Finished;
                            continue;
                        }
                        return Err("transcription requires more input or closure".into());
                    };
                    let mut text = String::new();
                    append_text(&mut self.decoder, block.tokens, &mut text)?;
                    if !text.is_empty() {
                        return Ok(Some(AudioTaskOutput::Text(text)));
                    }
                }
                State::Response(session) => match session.advance()? {
                    AudioAdvance::Refining => {}
                    AudioAdvance::Finished => self.state = State::Finished,
                    AudioAdvance::SpeechEnd(pcm) => {
                        if self.speech_only {
                            self.state = State::Finished;
                        }
                        if !pcm.is_empty() {
                            return Ok(Some(AudioTaskOutput::Pcm(pcm)));
                        }
                    }
                    AudioAdvance::Speech { pcm, .. } => {
                        if !pcm.is_empty() {
                            return Ok(Some(AudioTaskOutput::Pcm(pcm)));
                        }
                    }
                    AudioAdvance::Text { token, pcm_tail } => {
                        // A closing text decision may carry the last PCM samples.
                        let mut text = String::new();
                        append_text(&mut self.decoder, [token], &mut text)?;
                        if !pcm_tail.is_empty() {
                            // Retain any decoded text until the next advancement.
                            if !text.is_empty() {
                                self.pending_text = Some(text);
                            }
                            return Ok(Some(AudioTaskOutput::Pcm(pcm_tail)));
                        }
                        if !text.is_empty() {
                            return Ok(Some(AudioTaskOutput::Text(text)));
                        }
                    }
                },
                State::Ingress(_) => {
                    return Err("close the audio input before advancing the response".into())
                }
                State::Finished => {
                    self.decoder.finish()?;
                    return Ok(None);
                }
            }
        }
    }
}

fn append_text(
    decoder: &mut TextDecoder,
    ids: impl IntoIterator<Item = i32>,
    text: &mut String,
) -> TaskResult<()> {
    for id in ids {
        if let Some(fragment) = decoder.step(u32::try_from(id)?)? {
            text.push_str(&fragment);
        }
    }
    Ok(())
}
