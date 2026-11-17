//! Portable selection and startup for a finite Omni turn.

use std::collections::BTreeMap;

use eredu_checkpoint::recipe::RecipeDtype;
use eredu_core::{
    Audio, BackendFailure, Media, TokenizedMultimodalRequest, TokenizedMultimodalSegment,
};
use eredu_media::audio::AudioWaveform;
use eredu_nn::Tensor;
use eredu_runtime::{
    LayerWeightResidency, ParameterUnitMechanisms, PipelineTransportProvider,
    RealtimeGraphMechanisms, RealtimeMechanismSupport, RealtimeStartupMechanisms,
};

use super::{ArtifactError, MediaTokens, OmniVideoPrompt, PreparedOmni, ProcessorConfig};
use crate::{
    lfm3::{
        audio::{
            AudioMode, AudioParameterResidency, AudioPlacement, AudioRequest, AudioTaskResponse,
            LogMelFeatures, OmniAudioRequest, PreparedResidentAudio, ResidentAudioError,
        },
        vision::{PreparedVisionExecution, ProjectedVisionPrompt, VisionPromptAdmission},
    },
    processor_execution::{MediaGraphMechanisms, MediaInputLowering, PreparedProcessor},
};

/// Cold weight and language placement policy for an Omni speech response.
#[derive(Clone, Copy)]
pub struct OmniAudioSelection {
    /// Vision parameters are replicated on each response rank.
    pub vision: LayerWeightResidency,
    /// Independent speech graph parameter policies.
    pub speech: AudioParameterResidency,
    /// Language route. Distributed Omni uses the pipeline publication protocol.
    pub language: AudioPlacement,
}

impl Default for OmniAudioSelection {
    fn default() -> Self {
        Self {
            vision: LayerWeightResidency::FullyResident,
            speech: AudioParameterResidency::default(),
            language: AudioPlacement::Local,
        }
    }
}

/// Failure selecting or starting a prepared Omni turn.
#[derive(Debug, thiserror::Error)]
pub enum OmniTurnError {
    /// The requested policy is unavailable or the turn violates input constraints.
    #[error("Omni turn: {0}")]
    Selection(&'static str),
    /// The selected artifact cannot supply the requested weights.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// The retained weights or speech session could not be prepared.
    #[error(transparent)]
    Speech(#[from] ResidentAudioError),
    /// A vision operation or media placement failed.
    #[error(transparent)]
    Vision(#[from] eredu_nn::Error),
    /// Host audio transformation failed.
    #[error(transparent)]
    Media(#[from] eredu_media::MediaError),
    /// A native startup mechanism failed.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
}

/// Retains the exact vision and speech bindings and the ordered host request.
pub struct PreparedOmniTurn {
    speech: PreparedResidentAudio,
    vision: PreparedVisionExecution,
    vision_residency: LayerWeightResidency,
    media: PreparedProcessor,
    vision_admission: VisionPromptAdmission,
    prompt: TokenizedMultimodalRequest,
    video_prompt: OmniVideoPrompt,
    soundtracks: BTreeMap<usize, Audio>,
    processor: ProcessorConfig,
    markers: MediaTokens,
    audio_marker: i32,
    mode: AudioMode,
    max_slots: usize,
    seed: u64,
    output_rate: u32,
}

impl PreparedOmniTurn {
    /// Selects weights and admits an ordered, tokenized user turn without native allocation.
    pub fn select(
        artifact: &PreparedOmni,
        prompt: TokenizedMultimodalRequest,
        soundtracks: BTreeMap<usize, Audio>,
        mode: AudioMode,
        max_slots: usize,
        seed: u64,
        selection: OmniAudioSelection,
        support: &impl RealtimeMechanismSupport,
    ) -> Result<Self, OmniTurnError> {
        let facts = support.facts();
        if matches!(selection.language, AudioPlacement::Tensor(_)) {
            return Err(OmniTurnError::Selection(
                "distributed speech requires the stage publication route",
            ));
        }
        if !facts.supports_residency(selection.vision.execution_residency()) {
            return Err(OmniTurnError::Selection("vision parameter residency"));
        }
        let processor = artifact.speech().processor;
        let markers = artifact.media_tokens();
        let audio = artifact.speech().tokens;
        let reserved = [
            markers.image_token_id,
            markers.video_token_id,
            audio.input,
            audio.output,
        ];
        let mut video_count = 0;
        for part in prompt.segments() {
            match part {
                TokenizedMultimodalSegment::TokenIds(ids)
                    if ids.iter().any(|&id| reserved.contains(&(id as i32))) =>
                {
                    return Err(OmniTurnError::Selection(
                        "prompt contains an unbound media token",
                    ));
                }
                TokenizedMultimodalSegment::Media(Media::Video(video)) => {
                    if video.source_fps().is_none() {
                        return Err(OmniTurnError::Selection("video requires source FPS"));
                    }
                    video_count += 1;
                }
                _ => {}
            }
        }
        if soundtracks.keys().any(|&index| index >= video_count) {
            return Err(OmniTurnError::Selection(
                "soundtrack refers to a missing video",
            ));
        }
        if soundtracks
            .values()
            .any(|clip| clip.sample_rate() != processor.sample_rate())
        {
            return Err(OmniTurnError::Selection(
                "soundtrack must use the checkpoint sample rate",
            ));
        }
        if max_slots == 0 {
            return Err(OmniTurnError::Selection(
                "audio response budget must be positive",
            ));
        }

        let vision = artifact.prepare_vision(RecipeDtype::F32)?;
        vision.validate_device_window(selection.vision)?;
        let speech = PreparedResidentAudio::select_omni(
            artifact,
            selection.speech,
            selection.language,
            support,
        )?;
        Ok(Self {
            speech,
            vision,
            vision_residency: selection.vision,
            media: artifact.vision_processor(),
            vision_admission: artifact.vision_admission()?,
            prompt,
            video_prompt: artifact.video_prompt(),
            soundtracks,
            processor,
            markers,
            audio_marker: audio.input,
            mode,
            max_slots,
            seed,
            output_rate: u32::try_from(artifact.speech().codec.config().sampling_rate)
                .map_err(eredu_nn::Error::backend_source)?,
        })
    }

    /// Required sample rate of decoded PCM inputs.
    pub fn input_sample_rate(&self) -> u32 {
        self.processor.sample_rate()
    }

    /// Sample rate of generated PCM.
    pub const fn output_sample_rate(&self) -> u32 {
        self.output_rate
    }

    /// Binds the selected route, agrees rank-local media preparation, and starts speech prefill.
    pub fn start<M, E>(
        self,
        mechanisms: &M,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<Box<dyn AudioTaskResponse>, OmniTurnError>
    where
        M: MediaGraphMechanisms,
        E: std::error::Error + Send + Sync + 'static,
    {
        let graph = mechanisms.graph();
        let media_lowering = mechanisms.media_lowering();
        let context = graph.context();
        let Self {
            speech,
            vision,
            vision_residency,
            media,
            vision_admission,
            prompt,
            video_prompt,
            soundtracks,
            processor,
            markers,
            audio_marker,
            mode,
            max_slots,
            seed,
            ..
        } = self;
        let prepare = || {
            let mut vision =
                vision.bind::<M::Neural, _>(context, |source, addresses, units, build| {
                    graph
                        .components()
                        .bind_units(source, addresses, units, vision_residency, build)
                        .map(|(policy, _)| policy)
                })?;
            let projected = |request: &TokenizedMultimodalRequest| {
                let (input, identity) =
                    media_lowering.prepare_input(&media, request, encode_text)?;
                let input = vision_admission.admit_with_identity(
                    &input,
                    media_lowering.inspector(),
                    identity.as_ref(),
                )?;
                Ok(vision.project_prompt(input, context)?)
            };
            let prepared = prepare_prompt(
                prompt,
                soundtracks,
                processor,
                markers,
                audio_marker,
                video_prompt,
                projected,
                context,
            )?;
            Ok::<_, OmniTurnError>((prepared, graph.realize_random(seed)?))
        };
        if speech.placement().rank().is_some() {
            let (world, transport) = mechanisms.pipeline_transport().open()?;
            let mut model = speech.bind_pipeline(graph.clone(), world, transport)?;
            let (prepared, random) =
                agree_preparation(prepare(), |success| model.agree_prompt_preparation(success))?;
            Ok(Box::new(
                model.start_omni(prepared.request(mode, max_slots), random)?,
            ))
        } else {
            let (prepared, random) = prepare()?;
            let model = speech.bind_local(graph.clone())?;
            Ok(Box::new(
                model.start_omni(prepared.request(mode, max_slots), random)?,
            ))
        }
    }
}

fn agree_preparation<T>(
    prepared: Result<T, OmniTurnError>,
    agree: impl FnOnce(bool) -> Result<(), BackendFailure>,
) -> Result<T, OmniTurnError> {
    let agreement = agree(prepared.is_ok());
    let value = prepared?;
    agreement?;
    Ok(value)
}

struct PreparedPrompt<T> {
    tokens: Vec<i32>,
    features: Vec<LogMelFeatures>,
    image: Option<T>,
    video: Option<T>,
}

impl<T> PreparedPrompt<T> {
    fn request(&self, mode: AudioMode, max_slots: usize) -> OmniAudioRequest<'_, T> {
        OmniAudioRequest {
            audio: AudioRequest {
                tokens: &self.tokens,
                features: &self.features,
                replay: &[],
                mode,
                max_slots,
            },
            image: self.image.as_ref(),
            video: self.video.as_ref(),
        }
    }
}

fn prepare_prompt<T: Tensor>(
    prompt: TokenizedMultimodalRequest,
    soundtracks: BTreeMap<usize, Audio>,
    processor: ProcessorConfig,
    markers: MediaTokens,
    audio_marker: i32,
    video_prompt: OmniVideoPrompt,
    mut project: impl FnMut(
        &TokenizedMultimodalRequest,
    ) -> Result<ProjectedVisionPrompt<T>, OmniTurnError>,
    context: &T::Context,
) -> Result<PreparedPrompt<T>, OmniTurnError> {
    let transform = processor.transform()?;
    let mut features = Vec::new();
    let mut tokens = Vec::new();
    let mut images = Vec::new();
    let mut videos = Vec::new();
    let mut video_index = 0;
    for part in prompt.segments() {
        match part {
            TokenizedMultimodalSegment::TokenIds(ids) => {
                for &id in ids {
                    tokens.push(i32::try_from(id).map_err(eredu_nn::Error::backend_source)?);
                }
            }
            TokenizedMultimodalSegment::Media(Media::Audio(audio)) => {
                let clip =
                    transform.extract(AudioWaveform::new(audio.samples(), audio.sample_rate())?)?;
                let positions = processor.audio_positions(clip.frames)?;
                tokens.push(markers.vision_start_token_id);
                tokens.extend(std::iter::repeat_n(audio_marker, positions));
                tokens.push(markers.vision_end_token_id);
                features.push(clip);
            }
            TokenizedMultimodalSegment::Media(media) => {
                let input = TokenizedMultimodalRequest::from_media(media.clone());
                let (ids, image, video) = project(&input)?.into_media_parts();
                if let Media::Video(clip) = media {
                    let mut positions = 0;
                    if let Some(soundtrack) = soundtracks.get(&video_index) {
                        let audio = transform.extract(AudioWaveform::new(
                            soundtrack.samples(),
                            soundtrack.sample_rate(),
                        )?)?;
                        positions = processor.audio_positions(audio.frames)?;
                        features.push(audio);
                    }
                    tokens.extend(video_prompt.expand(clip, &ids, positions)?);
                    video_index += 1;
                } else {
                    tokens.extend(ids);
                }
                images.extend(image);
                videos.extend(video);
            }
        }
    }
    let combine = |mut rows: Vec<T>| match rows.len() {
        0 | 1 => Ok(rows.pop()),
        _ => T::concatenate(&rows, 0, context).map(Some),
    };
    Ok(PreparedPrompt {
        tokens,
        features,
        image: combine(images)?,
        video: combine(videos)?,
    })
}
