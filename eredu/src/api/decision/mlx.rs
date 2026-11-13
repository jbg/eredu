//! MLX realization of portable text decision requests.

use eredu_architectures::lfm3::{
    decision::{DecisionAudioProcessor, DecisionPrompt, PackedChoiceSet, TextChoicePreparation},
    Lfm3Input,
};
use eredu_backend_mlx::{
    native::{ExecutionContext, MlxVlDecoder},
    MlxTensor,
};
use eredu_core::BackendFailure;
use eredu_nn::Tensor;
use eredu_runtime::LayerWeightResidency;

use super::{
    ChoiceAnswer, ChoiceEvaluation, ChoiceQuestion, DecisionAnswer, DecisionDistribution,
    DecisionError, DecisionEvaluation, DecisionQuestion, PreparedTextChoiceModel,
};

#[cfg(feature = "image")]
mod visuals;

struct MediaProjection {
    rows: [Option<MlxTensor>; 3],
    audio: Option<DecisionAudioProcessor>,
    markers: eredu_architectures::lfm3::vision::MediaTokens,
    width: i32,
}

#[derive(Default)]
struct PreparedDecisionRows {
    rows: Vec<PackedChoiceSet>,
    media: Option<MediaProjection>,
    spoken_rows: Vec<Option<MlxTensor>>,
}

impl From<Vec<PackedChoiceSet>> for PreparedDecisionRows {
    fn from(rows: Vec<PackedChoiceSet>) -> Self {
        Self {
            rows,
            ..Self::default()
        }
    }
}

/// Error from preparing or advancing a choice evaluation.
#[derive(Debug, thiserror::Error)]
pub enum DecisionExecutionError {
    /// The request or its calibrated readout is invalid.
    #[error(transparent)]
    Decision(#[from] DecisionError),
    /// The selected backend could not execute the request.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// A rank rejected the prepared request, row, or readout.
    #[error("distributed decision agreement rejected the {0}")]
    Agreement(&'static str),
    /// The request was cancelled before its answers could be published.
    #[error("decision evaluation was cancelled")]
    Cancelled,
}

/// A loaded d1 text decision model.
pub struct TextChoiceModel {
    preparation: TextChoicePreparation,
    decoder: MlxVlDecoder,
    stream: <MlxTensor as Tensor>::Context,
    distributed: bool,
    #[cfg(feature = "image")]
    vision: Option<visuals::VisualRuntime>,
}

impl TextChoiceModel {
    /// Binds the retained d1 decoder using the selected weight residency.
    pub fn load(
        prepared: PreparedTextChoiceModel,
        residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder) = prepared.into_parts();
        Self::bind_decoder(preparation, decoder, residency, weights, execution)
    }

    fn bind_decoder(
        preparation: TextChoicePreparation,
        decoder: eredu_architectures::lfm3::vision::PreparedVlDecoder,
        residency: LayerWeightResidency,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let decoder = MlxVlDecoder::bind(decoder, residency, weights.stream(), execution.stream())?;
        Ok(Self::from_decoder(preparation, decoder, execution, false))
    }

    fn from_decoder(
        preparation: TextChoicePreparation,
        decoder: MlxVlDecoder,
        execution: &ExecutionContext,
        distributed: bool,
    ) -> Self {
        Self {
            preparation,
            decoder,
            stream: execution.stream().clone(),
            distributed,
            #[cfg(feature = "image")]
            vision: None,
        }
    }

    /// Loads one rank of a tensor, pipeline, or combined Ring decision model.
    ///
    /// Set `MLX_HOSTFILE` and `MLX_RANK` for every process. Supply the same
    /// prepared checkpoint, settings and request on all ranks, and drive their
    /// controlled sessions in lockstep. Retained media encoders are replicated.
    pub fn load_distributed(
        prepared: PreparedTextChoiceModel,
        distribution: DecisionDistribution,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<Self, DecisionExecutionError> {
        let (preparation, decoder, vision, audio) = prepared.into_execution_parts();
        let decoder = MlxVlDecoder::bind_ring(
            decoder,
            distribution.decoder_residency,
            distribution.topology,
            distribution.maximum_positions,
            weights.stream(),
            execution.stream(),
        )?;
        let mut model = Self::from_decoder(preparation, decoder, execution, true);
        #[cfg(feature = "image")]
        {
            let media = visuals::VisualRuntime::bind_optional(
                vision,
                audio,
                distribution,
                weights,
                execution,
            );
            model.vision = model.agree_result(
                [0x4445434c4f414400, 0, 0, 0, 0, 0, 0, 0],
                media,
                "media binding",
            )?;
        }
        #[cfg(not(feature = "image"))]
        {
            let media = if vision.is_none() && audio.is_none() {
                Ok(())
            } else {
                Err(DecisionError::Request("decision media requires the image feature").into())
            };
            model.agree_result(
                [0x4445434c4f414400, 0, 0, 0, 0, 0, 0, 0],
                media,
                "media binding",
            )?;
        }
        Ok(model)
    }

    fn agree_result<T>(
        &mut self,
        descriptor: [u64; 8],
        local: Result<T, DecisionExecutionError>,
        phase: &'static str,
    ) -> Result<T, DecisionExecutionError> {
        if !self.distributed {
            return local;
        }
        let agreed = self.decoder.agree_application(descriptor, local.is_ok());
        let value = local?;
        if !agreed? {
            return Err(DecisionExecutionError::Agreement(phase));
        }
        Ok(value)
    }

    /// Packs the input and starts a finite choice evaluation.
    pub fn start(
        &mut self,
        state: Option<&str>,
        questions: &[ChoiceQuestion],
    ) -> Result<ChoiceSession<'_>, DecisionExecutionError> {
        let rows = self
            .preparation
            .pack(state, questions)
            .map(Into::into)
            .map_err(Into::into);
        self.start_prepared(rows, questions.len(), PackedChoiceSet::choices)
    }

    /// Packs mixed question types and starts a finite decision evaluation.
    pub fn start_decisions(
        &mut self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionSession<'_>, DecisionExecutionError> {
        let rows = self
            .preparation
            .pack_decisions(state, questions)
            .map(Into::into)
            .map_err(Into::into);
        self.start_prepared(rows, questions.len(), PackedChoiceSet::decisions)
    }

    fn start_prepared<A>(
        &mut self,
        prepared: Result<PreparedDecisionRows, DecisionExecutionError>,
        questions: usize,
        readout: Readout<A>,
    ) -> Result<DecisionSession<'_, A>, DecisionExecutionError> {
        let mut descriptor = [0x4445435052455000, questions as u64, 0, 0, 0, 0, 0, 0];
        let input_tokens = prepared.as_ref().map_or(0, |p| {
            p.rows.iter().map(|row| row.tokens().len()).sum::<usize>()
        });
        if let Ok(prepared) = &prepared {
            descriptor[2] = prepared.rows.len() as u64;
            descriptor[3] = input_tokens as u64;
            for (word, chunk) in descriptor[4..]
                .iter_mut()
                .zip(self.preparation.calibration_sha256().chunks_exact(8))
            {
                *word = u64::from_le_bytes(chunk.try_into().expect("eight calibration bytes"));
            }
        }
        let prepared = self.agree_result(descriptor, prepared, "request preparation")?;
        Ok(DecisionSession {
            model: self,
            rows: prepared.rows.into_iter(),
            answers: Vec::with_capacity(questions),
            input_tokens,
            readout,
            finished: false,
            cancelled: false,
            next_row: 0,
            media: prepared.media,
            spoken_rows: prepared.spoken_rows.into_iter(),
        })
    }

    /// Evaluates all single-choice rows through the controlled session.
    pub fn evaluate(
        &mut self,
        state: Option<&str>,
        questions: &[ChoiceQuestion],
    ) -> Result<ChoiceEvaluation, DecisionExecutionError> {
        self.start(state, questions)?.run()
    }

    /// Evaluates mixed question types through the controlled session.
    pub fn evaluate_decisions(
        &mut self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEvaluation, DecisionExecutionError> {
        self.start_decisions(state, questions)?.run()
    }

    fn evaluate_row<A>(
        &mut self,
        row: &PackedChoiceSet,
        readout: Readout<A>,
        media: Option<&MediaProjection>,
        spoken: Option<&MlxTensor>,
    ) -> Result<Vec<A>, DecisionExecutionError> {
        let stream = self.stream.clone();
        let context = &stream;
        let prompt = match media {
            Some(media) => DecisionPrompt::<MlxTensor>::new_with_media(
                row,
                media.markers,
                media.audio,
                media.width,
                [
                    media.rows[0].as_ref(),
                    media.rows[1].as_ref(),
                    spoken.or(media.rows[2].as_ref()),
                ],
                context,
            ),
            None => DecisionPrompt::<MlxTensor>::new(row, context),
        }
        .map_err(BackendFailure::from_error);
        let local = prompt.and_then(|prompt| self.decoder.new_state().map(|state| (prompt, state)));
        let (prompt, mut state) = self.agree_result(
            [0x444543524f570000, 0, 0, 0, 0, 0, 0, 0],
            local.map_err(Into::into),
            "row preparation",
        )?;
        let output = self
            .decoder
            .forward(Lfm3Input::Decision(&prompt), &mut state)?;
        let result = (|| {
            let shape = output.shape();
            let [1, slots, width] = shape else {
                return Err(DecisionError::Request("expected batch-one answer slot logits").into());
            };
            if *slots as usize != row.answer_positions().len() || *width <= 0 {
                return Err(DecisionError::Request("unexpected answer slot dimensions").into());
            }
            let values = output
                .to_f32_vec(context)
                .map_err(BackendFailure::from_error)?;
            let logits = values.chunks_exact(*width as usize).collect::<Vec<_>>();
            Ok(readout(row, &logits)?)
        })();
        self.agree_result(
            [0x4445435245414400, 0, 0, 0, 0, 0, 0, 0],
            result,
            "calibrated readout",
        )
    }
}

type Readout<A> = fn(&PackedChoiceSet, &[&[f32]]) -> Result<Vec<A>, DecisionError>;

/// A controlled single-choice evaluation.
pub type ChoiceSession<'a> = DecisionSession<'a, ChoiceAnswer>;

/// A controlled evaluation that advances one complete packed row at a time.
pub struct DecisionSession<'a, A = DecisionAnswer> {
    model: &'a mut TextChoiceModel,
    rows: std::vec::IntoIter<PackedChoiceSet>,
    answers: Vec<A>,
    readout: Readout<A>,
    input_tokens: usize,
    finished: bool,
    cancelled: bool,
    next_row: usize,
    media: Option<MediaProjection>,
    spoken_rows: std::vec::IntoIter<Option<MlxTensor>>,
}

impl<A: serde::Serialize> DecisionSession<'_, A> {
    fn run(mut self) -> Result<DecisionEvaluation<A>, DecisionExecutionError> {
        loop {
            if let Some(evaluation) = self.advance()? {
                return Ok(evaluation);
            }
            if self.finished {
                return Err(DecisionExecutionError::Cancelled);
            }
        }
    }
    /// Runs one packed row and publishes answers only when all rows complete.
    pub fn advance(&mut self) -> Result<Option<DecisionEvaluation<A>>, DecisionExecutionError> {
        if self.finished {
            return Ok(None);
        }
        if self.model.distributed {
            let descriptor = [0x4445434144560000, self.next_row as u64, 0, 0, 0, 0, 0, 0];
            if let Err(error) = self
                .model
                .agree_result(descriptor, Ok(()), "row advancement")
            {
                self.finish_cancelled();
                return Err(error);
            }
            match self.model.decoder.agree_application(
                [0x44454343414e4300, self.next_row as u64, 0, 0, 0, 0, 0, 0],
                !self.cancelled,
            ) {
                Ok(true) => {}
                Ok(false) => {
                    self.finish_cancelled();
                    return Ok(None);
                }
                Err(error) => {
                    self.finish_cancelled();
                    return Err(error.into());
                }
            }
        }
        self.next_row += 1;
        let row = self
            .rows
            .next()
            .expect("unfinished choice session has a row");
        let spoken = self.spoken_rows.next().flatten();
        match self
            .model
            .evaluate_row(&row, self.readout, self.media.as_ref(), spoken.as_ref())
        {
            Ok(answers) => self.answers.extend(answers),
            Err(error) => {
                self.finish_cancelled();
                return Err(error);
            }
        }
        if !self.rows.as_slice().is_empty() {
            return Ok(None);
        }
        if self.model.distributed {
            use sha2::{Digest, Sha256};

            let mut descriptor = [
                0x4445435055420000,
                self.input_tokens as u64,
                self.answers.len() as u64,
                0,
                0,
                0,
                0,
                0,
            ];
            let digest = serde_json::to_vec(&self.answers)
                .map(Sha256::digest)
                .map_err(BackendFailure::from_error)
                .map_err(DecisionExecutionError::from);
            if let Ok(digest) = &digest {
                for (word, chunk) in descriptor[4..].iter_mut().zip(digest.chunks_exact(8)) {
                    *word = u64::from_le_bytes(chunk.try_into().expect("eight digest bytes"));
                }
            }
            if let Err(error) = self
                .model
                .agree_result(descriptor, digest, "answer publication")
            {
                self.finish_cancelled();
                return Err(error);
            }
        }
        self.finished = true;
        Ok(Some(DecisionEvaluation {
            answers: std::mem::take(&mut self.answers),
            input_tokens: self.input_tokens,
            generated_tokens: 0,
            calibration_sha256: self.model.preparation.calibration_sha256(),
        }))
    }

    /// Whether the evaluation has published, failed, or been cancelled.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Requests cancellation. A distributed session settles cancellation on the
    /// next collective `advance` call. Every rank must enter that call.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        if !self.model.distributed {
            self.finish_cancelled();
        }
    }

    fn finish_cancelled(&mut self) {
        self.finished = true;
        self.rows = Vec::new().into_iter();
        self.answers.clear();
        self.media = None;
        self.spoken_rows = Vec::new().into_iter();
    }
}
