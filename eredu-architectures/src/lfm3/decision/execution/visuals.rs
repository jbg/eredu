//! Visual ingress for the existing d1 decision session.

#[cfg(feature = "audio")]
use crate::lfm3::{
    audio::{AudioEncoder, EncoderModule, PolicyEncoder},
    decision::DecisionAudioProcessor,
};
use crate::lfm3::{
    decision::{PreparedDecisionAudio, PreparedDecisionVision},
    vision::{MediaTokens, PolicyVision, VisionModule, VisionPromptAdmission},
};
use crate::processor_execution::PreparedProcessor;
use eredu_core::input::MediaInputLowering;
use eredu_core::{Media, MultimodalRequest, MultimodalSegment, RgbImage};
use eredu_nn::Tensor;
use eredu_runtime::ParameterUnitMechanisms;

use super::{
    AttentionCache, BackendFailure, ComponentRankMechanisms, DecisionDistribution, DecisionError,
    DecisionEvaluation, DecisionExecutionError, DecisionQuestion, DecisionSession,
    DecoderPipelineState, DistributedNeuralBackend, LayerWeightResidency, LocalComponentMechanisms,
    MediaProjection, PackedChoiceSet, PreparedDecisionRows, PreparedTextChoiceModel,
    RuntimeStateComponents, TensorParallelGroupedNeuralBackend, TextChoiceModel,
};

#[cfg(feature = "audio")]
type SpeechEncoder<B, M> =
    PolicyEncoder<B, <M as ParameterUnitMechanisms<B>>::UnitPolicy<EncoderModule<B>>>;

pub(super) struct VisualRuntime<B, M>
where
    B: eredu_nn::NeuralBackend + 'static,
    B::Tensor: 'static,
    M: ParameterUnitMechanisms<B>,
{
    vision: PolicyVision<B, M::UnitPolicy<VisionModule<B>>>,
    processor: PreparedProcessor,
    admission: VisionPromptAdmission,
    markers: MediaTokens,
    max_pixels: u64,
    width: i32,
    #[cfg(feature = "audio")]
    audio: Option<(SpeechEncoder<B, M>, DecisionAudioProcessor)>,
}

impl<B, S, M, C, I> TextChoiceModel<B, S, M, C, I>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    M: LocalComponentMechanisms<B, S> + ComponentRankMechanisms<B, S, C>,
    <B::Tensor as Tensor>::Context: Clone,
    I: MediaInputLowering<B::Tensor>,
{
    /// Binds independently retained decoder and image/video encoder components.
    pub fn load_with_vision(
        prepared: PreparedTextChoiceModel,
        decoder_residency: LayerWeightResidency,
        vision_residency: LayerWeightResidency,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
        input_lowering: I,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder, vision) = prepared.into_image_parts()?;
        let mut model = Self::bind_decoder(
            preparation,
            decoder,
            decoder_residency,
            mechanisms,
            context,
            input_lowering,
        )?;
        model.vision = Some(VisualRuntime::bind(
            vision,
            vision_residency,
            mechanisms,
            context,
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
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
        input_lowering: I,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder, vision, audio) = prepared.into_omni_parts()?;
        let mut model = Self::bind_decoder(
            preparation,
            decoder,
            decoder_residency,
            mechanisms,
            context,
            input_lowering,
        )?;
        let mut vision = VisualRuntime::bind(vision, vision_residency, mechanisms, context)?;
        vision.bind_audio(audio, encoder_residency, mechanisms, context)?;
        model.vision = Some(vision);
        Ok(model)
    }

    /// Projects ordered images into the shared state before packing its questions.
    pub fn start_with_images(
        &mut self,
        state: Option<&str>,
        images: &[RgbImage],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_, B, S, M, C, I>, DecisionExecutionError> {
        let visuals = images.iter().cloned().map(Media::Image).collect::<Vec<_>>();
        self.start_with_visuals(state, &visuals, questions)
    }

    /// Projects ordered images and decoded video clips into the shared state.
    pub fn start_with_visuals(
        &mut self,
        state: Option<&str>,
        visuals: &[Media],
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_, B, S, M, C, I>, DecisionExecutionError> {
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
    ) -> Result<DecisionSession<'_, B, S, M, C, I>, DecisionExecutionError> {
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
    ) -> Result<DecisionSession<'_, B, S, M, C, I>, DecisionExecutionError> {
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
    ) -> Result<DecisionSession<'_, B, S, M, C, I>, DecisionExecutionError> {
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
    ) -> Result<PreparedDecisionRows<B::Tensor>, DecisionExecutionError> {
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
                let (ids, rows) = project_clip::<B, S, M>(encoder, processor, clip, &self.stream)?;
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
        let shared_audio = join_audio::<B::Tensor>(&projected, &self.stream)?;
        let voices = spoken
            .iter()
            .map(|clip| {
                clip.map(|clip| project_clip::<B, S, M>(encoder, processor, clip, &self.stream))
                    .transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (tokens, image, video) = if segments.is_empty() {
            (Vec::new(), None, None)
        } else {
            runtime.project(
                segments,
                &self.preparation,
                &self.stream,
                &self.input_lowering,
            )?
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
                spoken_rows.push(join_audio::<B::Tensor>(&clips, &self.stream)?);
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
    ) -> Result<PreparedDecisionRows<B::Tensor>, DecisionExecutionError> {
        let runtime = self.vision.as_mut().ok_or(DecisionError::Request(
            "decision model has no media binding",
        ))?;
        let segments =
            self.preparation
                .visual_segments(visuals, runtime.markers, runtime.max_pixels)?;
        let (tokens, image, video) = runtime.project(
            segments,
            &self.preparation,
            &self.stream,
            &self.input_lowering,
        )?;
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

impl<B, M> VisualRuntime<B, M>
where
    B: eredu_nn::NeuralBackend + 'static,
    B::Tensor: 'static,
    M: ParameterUnitMechanisms<B>,
{
    pub(super) fn bind_optional(
        vision: Option<PreparedDecisionVision>,
        audio: Option<PreparedDecisionAudio>,
        distribution: DecisionDistribution,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<Self>, DecisionExecutionError> {
        let vision = vision
            .map(|vision| Self::bind(vision, distribution.vision_residency, mechanisms, context))
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
                .bind_audio(audio, distribution.speech_residency, mechanisms, context)?;
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
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), DecisionExecutionError> {
        let (audio, processor) = audio.into_parts();
        let encoder = audio.bind(context, residency, mechanisms)?;
        self.audio = Some((encoder, processor));
        Ok(())
    }

    fn bind(
        prepared: PreparedDecisionVision,
        residency: LayerWeightResidency,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, DecisionExecutionError> {
        let (vision, processor, admission, max_pixels, markers, width) = prepared.into_parts();
        let vision = vision.bind(context, residency, mechanisms)?;
        Ok(Self {
            vision,
            processor,
            admission,
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
        preparation: &crate::lfm3::decision::TextChoicePreparation,
        stream: &<B::Tensor as Tensor>::Context,
        input_lowering: &impl MediaInputLowering<B::Tensor>,
    ) -> Result<(Vec<i32>, Option<B::Tensor>, Option<B::Tensor>), DecisionExecutionError> {
        let request = MultimodalRequest::new(segments).map_err(BackendFailure::from_error)?;
        let mut encode = |text: &str| preparation.encode_visual_text(text);
        let request = request
            .tokenize(&mut encode)
            .map_err(BackendFailure::from_error)?;
        let (prepared, identity) = eredu_runtime::processor_binding::prepare_graph_input(
            input_lowering,
            &self.processor,
            &request,
            &mut encode,
        )?;
        let prompt = self
            .admission
            .admit_with_identity(&prepared, input_lowering.inspector(), Some(&identity))
            .map_err(BackendFailure::from_error)?;
        Ok(self
            .vision
            .project_prompt(prompt, stream)
            .map_err(BackendFailure::from_error)?
            .into_media_parts())
    }
}

#[cfg(feature = "audio")]
fn project_clip<B, S, M>(
    encoder: &mut SpeechEncoder<B, M>,
    processor: DecisionAudioProcessor,
    clip: &eredu_core::Audio,
    stream: &<B::Tensor as Tensor>::Context,
) -> Result<(Vec<i32>, B::Tensor), DecisionExecutionError>
where
    B: eredu_nn::NeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    M: ParameterUnitMechanisms<B>,
{
    let (features, ids) = processor
        .prepare(clip)
        .map_err(BackendFailure::from_error)?;
    let rows = encoder.project_clip(&features, stream, |layout| {
        S::allocate_pipeline_state(layout.clone(), 0)
    })?;
    Ok((ids, rows))
}

#[cfg(feature = "audio")]
fn join_audio<T: Tensor>(clips: &[T], stream: &T::Context) -> Result<Option<T>, BackendFailure> {
    match clips {
        [] => Ok(None),
        [clip] => Ok(Some(clip.clone())),
        _ => T::concatenate(clips, 0, stream)
            .map(Some)
            .map_err(BackendFailure::from_error),
    }
}
