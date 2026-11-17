//! Finite multimodal requests through the shared turn-based speech session.

use std::collections::BTreeMap;

use eredu_architectures::lfm3::omni::PreparedOmniTurn;
use eredu_core::{Audio, Media, MediaBinding, MultimodalRequest, MultimodalSegment};
use serde_json::json;

use super::*;

pub use eredu_architectures::lfm3::omni::{OmniAudioSelection, PreparedOmni};

/// A finite mixed-media turn prepared with the selected Omni checkpoint.
pub struct PreparedMultimodalAudioTask {
    turn: PreparedOmniTurn,
    tokenizer: Tokenizer,
}

impl PreparedMultimodalAudioTask {
    /// Applies the checkpoint's chat template to one ordered user turn.
    ///
    /// Text segments carry user text. Image, video and mono PCM segments are
    /// decoded media. The waveform must use the checkpoint's input sample rate.
    pub fn new(
        artifact: &PreparedOmni,
        input: &MultimodalRequest,
        mode: AudioMode,
        max_slots: usize,
        seed: u64,
        support: &impl RealtimeMechanismSupport,
    ) -> TaskResult<Self> {
        Self::new_with_soundtracks(
            artifact,
            input,
            BTreeMap::new(),
            mode,
            max_slots,
            seed,
            OmniAudioSelection::default(),
            support,
        )
    }

    /// Pairs decoded PCM with videos by zero-based video order in the user turn.
    ///
    /// Each soundtrack starts at its video's time zero and uses the checkpoint sample rate.
    pub fn new_with_soundtracks(
        artifact: &PreparedOmni,
        input: &MultimodalRequest,
        soundtracks: BTreeMap<usize, Audio>,
        mode: AudioMode,
        max_slots: usize,
        seed: u64,
        selection: OmniAudioSelection,
        support: &impl RealtimeMechanismSupport,
    ) -> TaskResult<Self> {
        let assets = &artifact.speech().text_assets;
        let mut tokenizer = Tokenizer::from_bytes(&assets.tokenizer)?;
        let mut content = Vec::new();
        let mut bindings = Vec::new();
        for segment in input.segments() {
            content.push(match segment {
                MultimodalSegment::Text(text) => json!({"type":"text","text":text}),
                MultimodalSegment::TokenIds(_) => {
                    return Err(
                        "multimodal audio chat requires user text before tokenization".into(),
                    )
                }
                MultimodalSegment::Media(media) => {
                    bindings.push(MediaBinding::new(
                        artifact.placeholder(media),
                        media.clone(),
                    ));
                    let kind = match media {
                        Media::Image(_) => "image",
                        Media::Video(_) => "video",
                        Media::Audio(_) => "audio",
                    };
                    json!({"type":kind})
                }
            });
        }
        let rendered = tokenizer
            .apply_chat_template_json(
                ModelChatTemplate::Single(assets.chat_template.to_string()),
                [vec![json!({"role":"user","content":content})]],
                None,
                "turn-based-multimodal-audio",
                true,
                None,
            )?
            .pop()
            .ok_or("chat template returned no prompt")?;
        let prompt = MultimodalRequest::from_chat(&rendered, &bindings)?
            .tokenize(|text| encode(&tokenizer, text))?;
        let turn = PreparedOmniTurn::select(
            artifact,
            prompt,
            soundtracks,
            mode,
            max_slots,
            seed,
            selection,
            support,
        )?;
        Ok(Self { turn, tokenizer })
    }

    /// Required sample rate for decoded PCM inputs.
    pub fn input_sample_rate(&self) -> u32 {
        self.turn.input_sample_rate()
    }

    /// Sample rate of any generated mono PCM.
    pub fn output_sample_rate(&self) -> u32 {
        self.turn.output_sample_rate()
    }

    /// Binds resident weights, prepares the media and starts the shared response session.
    pub fn load(self, binding: &dyn AudioTaskBinding) -> TaskResult<AudioTaskSession> {
        let session = binding.bind(AudioTaskExecution::Omni {
            turn: Box::new(self.turn),
            encode_text: &mut |text| encode(&self.tokenizer, text),
        })?;
        let decoder = TextDecoder::from_shared_tokenizer(Arc::new((*self.tokenizer).clone()), true);
        Ok(AudioTaskSession::from_bound(session, decoder, None, false))
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, std::io::Error> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| std::io::Error::other(error.to_string()))
}
