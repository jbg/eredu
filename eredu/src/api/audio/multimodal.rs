//! Finite multimodal requests through the shared turn-based speech session.

use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::lfm3::{
    audio::{OmniAudioRequest, PreparedResidentAudio},
    vision::PreparedVisionExecution,
};
use eredu_backend_mlx::{
    native::{MlxPipelineAudioModel, MlxVlMedia, MlxVlVision},
    MlxTensor,
};
use eredu_checkpoint::recipe::RecipeDtype;
use eredu_core::{
    Audio, Media, MediaBinding, MultimodalRequest, MultimodalSegment, ParallelTopology,
    TokenizedMultimodalSegment,
};
use eredu_media::audio::AudioWaveform;
use eredu_nn::Tensor;
use eredu_runtime::LayerWeightResidency;

use super::*;

pub use eredu_architectures::lfm3::omni::PreparedOmni;

/// A finite mixed-media turn prepared with the selected Omni checkpoint.
pub struct PreparedMultimodalAudioTask {
    speech: PreparedResidentAudio,
    vision: PreparedVisionExecution,
    media: MlxVlMedia,
    prompt: eredu_core::TokenizedMultimodalRequest,
    video_prompt: eredu_architectures::lfm3::omni::OmniVideoPrompt,
    soundtracks: BTreeMap<usize, Audio>,
    tokenizer: Tokenizer,
    processor: eredu_architectures::lfm3::audio::ProcessorConfig,
    markers: eredu_architectures::lfm3::vision::MediaTokens,
    audio_marker: i32,
    mode: AudioMode,
    max_slots: usize,
    seed: u64,
    output_rate: u32,
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
    ) -> TaskResult<Self> {
        Self::new_with_soundtracks(artifact, input, BTreeMap::new(), mode, max_slots, seed)
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
    ) -> TaskResult<Self> {
        let videos = input.segments().iter().filter_map(|segment| match segment {
            MultimodalSegment::Media(Media::Video(video)) => Some(video),
            _ => None,
        });
        let mut video_count = 0;
        for video in videos {
            if video.source_fps().is_none() {
                return Err("Omni video requires source FPS".into());
            }
            video_count += 1;
        }
        if soundtracks.keys().any(|&index| index >= video_count) {
            return Err("Omni soundtrack refers to a missing video".into());
        }
        if soundtracks
            .values()
            .any(|clip| clip.sample_rate() != artifact.speech().processor.sample_rate())
        {
            return Err("Omni soundtrack must use the checkpoint sample rate".into());
        }
        if max_slots == 0 {
            return Err("audio response budget must be positive".into());
        }
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
        let markers = artifact.media_tokens();
        let audio = artifact.speech().tokens;
        for part in prompt.segments() {
            if let TokenizedMultimodalSegment::TokenIds(ids) = part {
                if ids.iter().any(|&id| {
                    [
                        markers.image_token_id,
                        markers.video_token_id,
                        audio.input,
                        audio.output,
                    ]
                    .contains(&(id as i32))
                }) {
                    return Err("the prompt contains an unbound media token".into());
                }
            }
        }
        Ok(Self {
            speech: PreparedResidentAudio::select_omni(artifact)?,
            vision: artifact.prepare_vision(RecipeDtype::F32)?,
            media: MlxVlMedia::for_omni(artifact)?,
            prompt,
            video_prompt: artifact.video_prompt(),
            soundtracks,
            tokenizer,
            processor: artifact.speech().processor,
            markers,
            audio_marker: audio.input,
            mode,
            max_slots,
            seed,
            output_rate: u32::try_from(artifact.speech().codec.config().sampling_rate)?,
        })
    }

    /// Required sample rate for decoded PCM inputs.
    pub fn input_sample_rate(&self) -> u32 {
        self.processor.sample_rate()
    }

    /// Sample rate of any generated mono PCM.
    pub fn output_sample_rate(&self) -> u32 {
        self.output_rate
    }

    /// Binds resident weights, prepares the media and starts the shared response session.
    pub fn load(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> TaskResult<AudioTaskSession> {
        self.load_with_residency(
            weights,
            execution,
            LayerWeightResidency::FullyResident,
            AudioParameterResidency::default(),
        )
    }

    /// Applies independent vision and speech parameter policies to the selected components.
    pub fn load_with_residency(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
        vision_residency: LayerWeightResidency,
        speech_residency: AudioParameterResidency,
    ) -> TaskResult<AudioTaskSession> {
        self.load_response(weights, execution, vision_residency, speech_residency, None)
    }

    /// Binds a tensor, pipeline, or combined rank using the MLX TCP Ring transport.
    ///
    /// Set `MLX_HOSTFILE` and `MLX_RANK` before starting each process. All ranks
    /// must supply the same task, seed and topology and drain their output in
    /// lockstep. Vision and speech components are replicated on each rank.
    pub fn load_distributed(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
        topology: ParallelTopology,
        vision_residency: LayerWeightResidency,
        speech_residency: AudioParameterResidency,
    ) -> TaskResult<AudioTaskSession> {
        self.load_response(
            weights,
            execution,
            vision_residency,
            speech_residency,
            Some(topology),
        )
    }

    fn load_response(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
        vision_residency: LayerWeightResidency,
        speech_residency: AudioParameterResidency,
        topology: Option<ParallelTopology>,
    ) -> TaskResult<AudioTaskSession> {
        enum Model {
            Resident(PreparedResidentAudio),
            Distributed(MlxPipelineAudioModel),
        }
        let mut model = match topology {
            Some(topology) => Model::Distributed(MlxPipelineAudioModel::load_ring(
                self.speech,
                speech_residency,
                topology,
                weights,
                execution,
            )?),
            None => Model::Resident(self.speech),
        };
        let media = (|| -> TaskResult<_> {
            let mut vision = MlxVlVision::bind(
                self.vision,
                vision_residency,
                weights.stream(),
                execution.stream(),
            )?;
            let transform = self.processor.transform()?;
            let mut tokens = Vec::new();
            let mut images = Vec::new();
            let mut videos = Vec::new();
            let mut features = Vec::new();
            let mut video_index = 0;
            for part in self.prompt.segments() {
                match part {
                    TokenizedMultimodalSegment::TokenIds(ids) => {
                        tokens.extend(
                            ids.iter()
                                .copied()
                                .map(i32::try_from)
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    TokenizedMultimodalSegment::Media(Media::Audio(audio)) => {
                        let clip = transform
                            .extract(AudioWaveform::new(audio.samples(), audio.sample_rate())?)?;
                        let positions = self.processor.audio_positions(clip.frames)?;
                        tokens.push(self.markers.vision_start_token_id);
                        tokens.extend(std::iter::repeat_n(self.audio_marker, positions));
                        tokens.push(self.markers.vision_end_token_id);
                        features.push(clip);
                    }
                    TokenizedMultimodalSegment::Media(media) => {
                        let input =
                            MultimodalRequest::new(vec![MultimodalSegment::Media(media.clone())])?
                                .tokenize(|text| encode(&self.tokenizer, text))?;
                        let prepared = self
                            .media
                            .prepare(&input, &mut |text| encode(&self.tokenizer, text))?;
                        let (ids, image, video) = vision
                            .project_prompt(self.media.admit(&prepared)?)?
                            .into_media_parts();
                        if let Media::Video(clip) = media {
                            let mut positions = 0;
                            if let Some(soundtrack) = self.soundtracks.get(&video_index) {
                                let audio = transform.extract(AudioWaveform::new(
                                    soundtrack.samples(),
                                    soundtrack.sample_rate(),
                                )?)?;
                                positions = self.processor.audio_positions(audio.frames)?;
                                features.push(audio);
                            }
                            tokens.extend(self.video_prompt.expand(clip, &ids, positions)?);
                            video_index += 1;
                        } else {
                            tokens.extend(ids);
                        }
                        images.extend(image);
                        videos.extend(video);
                    }
                }
            }
            let combine = |mut rows: Vec<MlxTensor>| match rows.len() {
                0 | 1 => Ok(rows.pop()),
                _ => MlxTensor::concatenate(&rows, 0, execution.stream()).map(Some),
            };
            let image = combine(images)?;
            let video = combine(videos)?;
            Ok((tokens, features, image, video))
        })();
        let agreement = match &mut model {
            Model::Distributed(model) => model.agree_prompt_preparation(media.is_ok()),
            Model::Resident(_) => Ok(()),
        };
        let (tokens, features, image, video) = media?;
        agreement?;
        let request = OmniAudioRequest {
            audio: AudioRequest {
                tokens: &tokens,
                features: &features,
                replay: &[],
                mode: self.mode,
                max_slots: self.max_slots,
            },
            image: image.as_ref(),
            video: video.as_ref(),
        };
        let session = match model {
            Model::Distributed(model) => {
                ResponseSession::Distributed(model.start_omni(request, self.seed)?)
            }
            Model::Resident(speech) => {
                let model = MlxAudioModel::load_prepared_with_parameter_residency(
                    speech,
                    speech_residency,
                    weights,
                    execution,
                )?;
                ResponseSession::Resident(model.start_omni(request, self.seed)?)
            }
        };
        Ok(AudioTaskSession {
            state: State::Response(session),
            decoder: TextDecoder::from_shared_tokenizer(Arc::new((*self.tokenizer).clone()), true),
            input_rate: None,
            pending_text: None,
            speech_only: false,
        })
    }
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, std::io::Error> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| std::io::Error::other(error.to_string()))
}
