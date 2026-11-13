//! LFM3 Audio chat and generation policy for turn-based requests.

use serde_json::{json, Value};

use super::{AudioMode, TextAssets};

/// Turn-based tasks that use the LFM3 Audio chat template.
pub enum AudioTurn<'a> {
    /// Answer a spoken request with interleaved text and speech.
    Respond {
        /// Optional instruction accompanying the audio.
        instruction: &'a str,
    },
    /// Translate speech into text.
    Translate {
        /// Language spoken in the input.
        source_language: &'a str,
        /// Language of the translated text.
        language: &'a str,
    },
    /// Synthesize speech from text.
    Speak {
        /// Text to synthesize.
        text: &'a str,
    },
}

/// Family-specific inputs to the facade's chat-template renderer and session.
pub struct AudioTurnPrompt<'a> {
    /// Template selected from the checkpoint's text assets.
    pub template: &'a str,
    /// Messages to render with an assistant generation prompt.
    pub messages: Vec<Value>,
    /// Generation mode that supplies the assistant's opening marker.
    pub mode: AudioMode,
}

impl AudioTurn<'_> {
    /// Selects the model's chat format and generation mode for this turn.
    pub fn prompt<'a>(self, assets: &'a TextAssets) -> AudioTurnPrompt<'a> {
        let (content, system, mode, named_template) = match self {
            Self::Respond { instruction } => {
                let mut content = vec![json!({"type": "audio"})];
                if !instruction.trim().is_empty() {
                    content.push(json!({"type": "text", "text": instruction}));
                }
                (
                    json!(content),
                    "Respond with interleaved text and audio.".into(),
                    AudioMode::Interleaved,
                    None,
                )
            }
            Self::Translate {
                source_language,
                language,
            } => (
                json!([{"type": "audio"}]),
                format!("Translate the following speech from {source_language} to {language} in text form."),
                AudioMode::Text,
                None,
            ),
            Self::Speak { text } => (
                json!(text),
                "Perform TTS in American English using a female voice.".into(),
                AudioMode::Speech,
                assets.named_chat_templates.get("tts"),
            ),
        };
        let mut messages = Vec::new();
        if named_template.is_none() {
            messages.push(json!({"role": "system", "content": system}));
        }
        messages.push(json!({"role": "user", "content": content}));
        AudioTurnPrompt {
            template: named_template.unwrap_or(&assets.chat_template),
            messages,
            mode,
        }
    }
}
