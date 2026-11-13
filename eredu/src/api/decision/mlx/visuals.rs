//! Visual ingress for the existing d1 decision session.

#[cfg(feature = "audio")]
use eredu_architectures::lfm3::decision::DecisionAudioProcessor;
use eredu_architectures::lfm3::{
    decision::{PreparedDecisionAudio, PreparedDecisionVision},
    vision::MediaTokens,
};
use eredu_backend_mlx::native::{MlxVlMedia, MlxVlVision};
#[cfg(feature = "audio")]
use eredu_backend_mlx::{native::MlxAudioEncoder, MlxTensor};
use eredu_core::{Media, MultimodalRequest, MultimodalSegment, RgbImage};
#[cfg(feature = "audio")]
use eredu_nn::Tensor;

use super::{
    BackendFailure, DecisionDistribution, DecisionError, DecisionEvaluation,
    DecisionExecutionError, DecisionQuestion, DecisionSession, ExecutionContext,
    LayerWeightResidency, MediaProjection, PackedChoiceSet, PreparedDecisionRows,
    PreparedTextChoiceModel, TextChoiceModel,
};

pub(super) struct VisualRuntime {
    vision: MlxVlVision,
    processor: MlxVlMedia,
    markers: MediaTokens,
    max_pixels: u64,
    width: i32,
    #[cfg(feature = "audio")]
    audio: Option<(MlxAudioEncoder, DecisionAudioProcessor)>,
}

impl TextChoiceModel {
    /// Binds the independently retained decoder and vision components.
    pub fn load_with_images(
        prepared: PreparedTextChoiceModel,
        decoder_residency: LayerWeightResidency,
        vision_residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        Self::load_with_vision(
            prepared,
            decoder_residency,
            vision_residency,
            weights,
            execution,
        )
    }

    /// Binds independently retained decoder and image/video encoder components.
    pub fn load_with_vision(
        prepared: PreparedTextChoiceModel,
        decoder_residency: LayerWeightResidency,
        vision_residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder, vision) = prepared.into_image_parts()?;
        let mut model =
            Self::bind_decoder(preparation, decoder, decoder_residency, weights, execution)?;
        model.vision = Some(VisualRuntime::bind(
            vision,
            vision_residency,
            weights,
            execution,
        )?);
        Ok(model)
    }

    /// Binds d1 Omni's decoder, vision, and speech input with independent residency policies.
    #[cfg(feature = "audio")]
    pub fn load_with_omni(
        prepared: PreparedTextChoiceModel,
        decoder_residency: LayerWeightResidency,
        vision_residency: LayerWeightResidency,
        encoder_residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder, vision, audio) = prepared.into_omni_parts()?;
        let mut model =
            Self::bind_decoder(preparation, decoder, decoder_residency, weights, execution)?;
        let mut vision = VisualRuntime::bind(vision, vision_residency, weights, execution)?;
        vision.bind_audio(audio, encoder_residency, weights, execution)?;
        model.vision = Some(vision);
        Ok(model)
    }

    /// Projects ordered images into the shared state before packing its questions.
    pub fn start_with_images(
        &mut self,
        state: Option<&str>,
        images: &[RgbImage],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        let visuals = images.iter().cloned().map(Media::Image).collect::<Vec<_>>();
        self.start_with_visuals(state, &visuals, questions)
    }

    /// Projects ordered images and decoded video clips into the shared state.
    pub fn start_with_visuals(
        &mut self,
        state: Option<&str>,
        visuals: &[Media],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        if visuals.is_empty() {
            return self.start_decisions(state, questions);
        }
        let prepared = self.prepare_visuals(state, questions, visuals);
        self.start_prepared(prepared, questions.len(), PackedChoiceSet::decisions)
    }

    /// Projects ordered image, video and mono PCM state through the decision session.
    #[cfg(feature = "audio")]
    pub fn start_with_media(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        if media.iter().all(|item| !matches!(item, Media::Audio(_))) {
            return self.start_with_visuals(state, media, questions);
        }
        self.start_omni(state, media, questions, &[])
    }

    /// Uses one optional spoken instruction clip per question. A clip replaces that
    /// question's written instructions while retaining its typed answer settings.
    #[cfg(feature = "audio")]
    pub fn start_with_spoken_questions(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
        spoken: &[Option<&eredu_core::Audio>],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        if spoken.len() != questions.len() {
            return self.start_prepared(
                Err(DecisionError::Request("spoken instructions must align with questions").into()),
                questions.len(),
                PackedChoiceSet::decisions,
            );
        }
        if spoken.iter().all(Option::is_none) {
            return self.start_with_media(state, media, questions);
        }
        self.start_omni(state, media, questions, spoken)
    }

    /// Evaluates written and spoken questions through the controlled decision session.
    #[cfg(feature = "audio")]
    pub fn evaluate_with_spoken_questions(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
        spoken: &[Option<&eredu_core::Audio>],
    ) -> Result<DecisionEvaluation, DecisionExecutionError> {
        self.start_with_spoken_questions(state, media, questions, spoken)?
            .run()
    }

    #[cfg(feature = "audio")]
    fn start_omni(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
        spoken: &[Option<&eredu_core::Audio>],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        let prepared = self.prepare_omni(state, media, questions, spoken);
        self.start_prepared(prepared, questions.len(), PackedChoiceSet::decisions)
    }

    #[cfg(feature = "audio")]
    fn prepare_omni(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
        spoken: &[Option<&eredu_core::Audio>],
    ) -> Result<PreparedDecisionRows, DecisionExecutionError> {
        let runtime = self.vision.as_mut().ok_or(DecisionError::Request(
            "decision model has no media binding",
        ))?;
        let (encoder, processor) = runtime.audio.as_mut().ok_or(DecisionError::Request(
            "decision model has no audio binding",
        ))?;
        let processor = *processor;
        let mut segments = Vec::new();
        let mut projected = Vec::new();
        for item in media {
            if let Media::Audio(clip) = item {
                let (ids, rows) = project_clip(encoder, processor, clip, &self.stream)?;
                projected.push(rows);
                segments.push(MultimodalSegment::TokenIds(
                    ids.into_iter().map(|id| id as u32).collect(),
                ));
            } else {
                segments.extend(self.preparation.visual_segments(
                    std::slice::from_ref(item),
                    runtime.markers,
                    runtime.max_pixels,
                )?);
            }
        }
        let shared_audio = join_audio(&projected, &self.stream)?;
        let voices = spoken
            .iter()
            .map(|clip| {
                clip.map(|clip| project_clip(encoder, processor, clip, &self.stream))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (tokens, image, video) = if segments.is_empty() {
            (Vec::new(), None, None)
        } else {
            runtime.project(segments, &self.preparation)?
        };
        let rows = if voices.is_empty() {
            self.preparation.pack_media_decisions(
                state,
                questions,
                &tokens,
                runtime.markers,
                Some(processor),
            )?
        } else {
            let questions = questions
                .iter()
                .zip(
                    voices
                        .iter()
                        .map(|voice| voice.as_ref().map(|(ids, _)| ids.as_slice())),
                )
                .collect::<Vec<_>>();
            self.preparation.pack_spoken_decisions(
                state,
                &questions,
                &tokens,
                runtime.markers,
                processor,
            )?
        };
        let mut spoken_rows = Vec::new();
        if !voices.is_empty() {
            let mut voices = voices.into_iter();
            for row in &rows {
                let mut clips = shared_audio.iter().cloned().collect::<Vec<_>>();
                clips.extend(
                    voices
                        .by_ref()
                        .take(row.question_count())
                        .flatten()
                        .map(|(_, rows)| rows),
                );
                spoken_rows.push(join_audio(&clips, &self.stream)?);
            }
        }
        let projection = MediaProjection {
            rows: [image, video, shared_audio],
            audio: Some(processor),
            markers: runtime.markers,
            width: runtime.width,
        };
        Ok(PreparedDecisionRows {
            rows,
            media: Some(projection),
            spoken_rows,
        })
    }

    /// Evaluates mixed state through the controlled decision session.
    #[cfg(feature = "audio")]
    pub fn evaluate_with_media(
        &mut self,
        state: Option<&str>,
        media: &[Media],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEvaluation, DecisionExecutionError> {
        self.start_with_media(state, media, questions)?.run()
    }

    fn prepare_visuals(
        &mut self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
        visuals: &[Media],
    ) -> Result<PreparedDecisionRows, DecisionExecutionError> {
        let runtime = self.vision.as_mut().ok_or(DecisionError::Request(
            "decision model has no media binding",
        ))?;
        let segments =
            self.preparation
                .visual_segments(visuals, runtime.markers, runtime.max_pixels)?;
        let (tokens, image, video) = runtime.project(segments, &self.preparation)?;
        let rows =
            self.preparation
                .pack_visual_decisions(state, questions, &tokens, runtime.markers)?;
        let projection = MediaProjection {
            rows: [image, video, None],
            audio: None,
            markers: runtime.markers,
            width: runtime.width,
        };
        Ok(PreparedDecisionRows {
            rows,
            media: Some(projection),
            ..PreparedDecisionRows::default()
        })
    }

    /// Evaluates image and video state through the controlled decision session.
    pub fn evaluate_with_visuals(
        &mut self,
        state: Option<&str>,
        visuals: &[Media],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEvaluation, DecisionExecutionError> {
        self.start_with_visuals(state, visuals, questions)?.run()
    }

    /// Evaluates image-conditioned questions through the controlled decision session.
    pub fn evaluate_with_images(
        &mut self,
        state: Option<&str>,
        images: &[RgbImage],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEvaluation, DecisionExecutionError> {
        self.start_with_images(state, images, questions)?.run()
    }
}

impl VisualRuntime {
    pub(super) fn bind_optional(
        vision: Option<PreparedDecisionVision>,
        audio: Option<PreparedDecisionAudio>,
        distribution: DecisionDistribution,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Option<Self>, DecisionExecutionError> {
        let vision = vision
            .map(|vision| Self::bind(vision, distribution.vision_residency, weights, execution))
            .transpose()?;
        #[cfg(feature = "audio")]
        let mut vision = vision;
        if let Some(audio) = audio {
            #[cfg(feature = "audio")]
            vision
                .as_mut()
                .ok_or(DecisionError::Request(
                    "decision audio requires a media binding",
                ))?
                .bind_audio(audio, distribution.speech_residency, weights, execution)?;
            #[cfg(not(feature = "audio"))]
            {
                drop(audio);
                return Err(DecisionError::Request(
                    "decision speech input requires the audio feature",
                )
                .into());
            }
        }
        Ok(vision)
    }

    #[cfg(feature = "audio")]
    fn bind_audio(
        &mut self,
        audio: PreparedDecisionAudio,
        residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<(), DecisionExecutionError> {
        let (audio, processor) = audio.into_parts();
        self.audio = Some((
            MlxAudioEncoder::bind(audio, residency, weights.stream(), execution.stream())?,
            processor,
        ));
        Ok(())
    }

    fn bind(
        prepared: PreparedDecisionVision,
        residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let (vision, processor, admission, max_pixels, markers, width) = prepared.into_parts();
        let vision = MlxVlVision::bind(vision, residency, weights.stream(), execution.stream())?;
        Ok(Self {
            vision,
            processor: MlxVlMedia::from_prepared(processor, admission),
            markers,
            max_pixels,
            width,
            #[cfg(feature = "audio")]
            audio: None,
        })
    }

    fn project(
        &mut self,
        segments: Vec<MultimodalSegment>,
        preparation: &eredu_architectures::lfm3::decision::TextChoicePreparation,
    ) -> Result<
        (
            Vec<i32>,
            Option<eredu_backend_mlx::MlxTensor>,
            Option<eredu_backend_mlx::MlxTensor>,
        ),
        DecisionExecutionError,
    > {
        let request = MultimodalRequest::new(segments).map_err(BackendFailure::from_error)?;
        let mut encode = |text: &str| preparation.encode_visual_text(text);
        let request = request
            .tokenize(&mut encode)
            .map_err(BackendFailure::from_error)?;
        let prepared = self
            .processor
            .prepare(&request, &mut encode)
            .map_err(BackendFailure::from_error)?;
        Ok(self
            .vision
            .project_prompt(self.processor.admit(&prepared)?)?
            .into_media_parts())
    }
}

#[cfg(feature = "audio")]
fn project_clip(
    encoder: &mut MlxAudioEncoder,
    processor: DecisionAudioProcessor,
    clip: &eredu_core::Audio,
    stream: &<MlxTensor as Tensor>::Context,
) -> Result<(Vec<i32>, MlxTensor), DecisionExecutionError> {
    let (features, ids) = processor
        .prepare(clip)
        .map_err(BackendFailure::from_error)?;
    let frames = i32::try_from(features.frames).map_err(BackendFailure::from_error)?;
    let bins = i32::try_from(features.mel_bins).map_err(BackendFailure::from_error)?;
    let tensor = MlxTensor::from_f32_slice(&features.values, &[1, frames, bins], stream)
        .and_then(|features| features.transpose_axes(&[0, 2, 1], stream))
        .map_err(BackendFailure::from_error)?;
    let rows = encoder
        .project(&tensor)?
        .squeeze_axes(&[0], stream)
        .map_err(BackendFailure::from_error)?;
    Ok((ids, rows))
}

#[cfg(feature = "audio")]
fn join_audio(
    clips: &[MlxTensor],
    stream: &<MlxTensor as Tensor>::Context,
) -> Result<Option<MlxTensor>, BackendFailure> {
    match clips {
        [] => Ok(None),
        [clip] => Ok(Some(clip.clone())),
        _ => MlxTensor::concatenate(clips, 0, stream)
            .map(Some)
            .map_err(BackendFailure::from_error),
    }
}
