//! Backend-neutral execution and observation of realtime traces.

use eredu_core::{
    ObservationSet, ObservationValue, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
    RealtimeSpeechConfig, TensorObservation, TensorObservationData,
};
use eredu_nn::Tensor;
use std::error::Error;

use crate::{observe_i32_tensor, EvidenceError};

/// Evaluation-owned execution seam for one portable realtime trace.
///
/// Implementations are composition adapters: they may drive a native or
/// reference executable, but evaluation only observes portable frames and
/// sampling controls. In particular, this contract does not make a concrete
/// tensor backend responsible for model, session, or scheduling policy.
pub trait RealtimeEvaluationDriver {
    /// Driver-specific execution failure.
    type Error: Error + Send + Sync + 'static;

    /// Portable token geometry used by this executable.
    fn speech_config(&self) -> &RealtimeSpeechConfig;

    /// Starts a fresh request-local trace with the supplied sampling controls.
    fn start_trace(&mut self, sampling: RealtimeSampling) -> Result<(), Self::Error>;

    /// Executes and observes one portable input frame.
    fn evaluate_frame(
        &mut self,
        frame: RealtimeInputFrame,
    ) -> Result<RealtimeOutputFrame, Self::Error>;

    /// Finishes the active trace and releases its request-local state.
    fn finish_trace(&mut self) -> Result<(), Self::Error>;
}

/// Completed portable outputs from one realtime request.
#[derive(Debug, Clone)]
pub struct RealtimeTrace {
    batch: usize,
    generated_audio_codebooks: usize,
    frames: Vec<RealtimeOutputFrame>,
}

impl RealtimeTrace {
    /// Stable batch dimension.
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Generated audio codebooks per output frame.
    pub const fn generated_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks
    }

    /// Completed frames in submission order.
    pub fn frames(&self) -> &[RealtimeOutputFrame] {
        &self.frames
    }

    /// Converts common token streams and per-decision diagnostics to evidence.
    pub fn observations(&self) -> Result<ObservationSet, RealtimeTraceError> {
        let mut observations = ObservationSet::new();
        observations.insert(
            "trace.text_tokens",
            ObservationValue::Tensor(integer_tensor(
                vec![self.batch, self.frames.len()],
                transpose_frame_values(
                    self.frames.iter().map(RealtimeOutputFrame::text_tokens),
                    self.batch,
                    1,
                )?,
            )?),
        )?;
        observations.insert(
            "trace.sampled_audio_tokens",
            ObservationValue::Tensor(integer_tensor(
                vec![
                    self.batch,
                    self.generated_audio_codebooks,
                    self.frames.len(),
                ],
                transpose_frame_values(
                    self.frames
                        .iter()
                        .map(RealtimeOutputFrame::sampled_audio_tokens),
                    self.batch,
                    self.generated_audio_codebooks,
                )?,
            )?),
        )?;
        let emitted = self
            .frames
            .iter()
            .filter_map(RealtimeOutputFrame::output_audio_tokens)
            .collect::<Vec<_>>();
        observations.insert(
            "trace.output_audio_tokens",
            ObservationValue::Tensor(integer_tensor(
                vec![self.batch, self.generated_audio_codebooks, emitted.len()],
                transpose_frame_values(
                    emitted.iter().copied(),
                    self.batch,
                    self.generated_audio_codebooks,
                )?,
            )?),
        )?;
        for (frame, output) in self.frames.iter().enumerate() {
            for diagnostic in output.diagnostics() {
                observations.insert(
                    format!(
                        "frames.{frame}.decisions.{}.logits",
                        diagnostic.prediction()
                    ),
                    ObservationValue::Tensor(diagnostic.tensor().clone()),
                )?;
            }
        }
        Ok(observations)
    }

    /// Stacks text followed by sampled-audio tokens as `[batch, width, frames]`.
    pub fn combined_sampled_tokens(
        &self,
        skip_frames: usize,
    ) -> Result<TensorObservation, RealtimeTraceError> {
        let frames = self.frames.get(skip_frames..).unwrap_or_default();
        let width = self.generated_audio_codebooks + 1;
        let mut values = Vec::with_capacity(self.batch * width * frames.len());
        for batch_index in 0..self.batch {
            for value_index in 0..width {
                for frame in frames {
                    let value = if value_index == 0 {
                        *frame.text_tokens().get(batch_index).ok_or(
                            RealtimeTraceError::FrameWidth {
                                batch: self.batch,
                                width: 1,
                                values: frame.text_tokens().len(),
                            },
                        )?
                    } else {
                        *frame
                            .sampled_audio_tokens()
                            .get(batch_index * self.generated_audio_codebooks + value_index - 1)
                            .ok_or(RealtimeTraceError::FrameWidth {
                                batch: self.batch,
                                width: self.generated_audio_codebooks,
                                values: frame.sampled_audio_tokens().len(),
                            })?
                    };
                    values.push(i64::from(value));
                }
            }
        }
        integer_tensor(vec![self.batch, width, frames.len()], values)
    }

    /// Frame indices at which delay-aligned output audio was emitted.
    pub fn emitted_frame_indices(&self) -> Result<TensorObservation, RealtimeTraceError> {
        let values = self
            .frames
            .iter()
            .enumerate()
            .filter_map(|(index, frame)| frame.output_audio_tokens().is_some().then_some(index))
            .map(|index| i64::try_from(index).map_err(|_| RealtimeTraceError::IndexOverflow(index)))
            .collect::<Result<Vec<_>, _>>()?;
        integer_tensor(vec![values.len()], values)
    }
}

/// Converts neutral `[batch, codebooks, frames]` integer tokens to frame inputs.
pub fn encoded_audio_frames<T: Tensor>(
    tokens: &T,
    context: &T::Context,
) -> Result<Vec<RealtimeInputFrame>, RealtimeTraceError> {
    let observed = observe_i32_tensor(tokens, context)?;
    let [batch, codebooks, frames] = observed.shape() else {
        return Err(RealtimeTraceError::InputShape(observed.shape().to_vec()));
    };
    let TensorObservationData::I64(values) = observed.data() else {
        unreachable!("observe_i32_tensor always produces I64 host values")
    };
    (0..*frames)
        .map(|frame| {
            let mut frame_tokens = Vec::with_capacity(batch * codebooks);
            for batch_index in 0..*batch {
                for codebook in 0..*codebooks {
                    let index = (batch_index * codebooks + codebook) * frames + frame;
                    frame_tokens.push(
                        i32::try_from(values[index])
                            .map_err(|_| RealtimeTraceError::TokenRange(values[index]))?,
                    );
                }
            }
            Ok(RealtimeInputFrame::new(*batch, frame_tokens))
        })
        .collect()
}

/// Executes portable encoded frames through an evaluation driver.
pub fn run_realtime_trace<D>(
    driver: &mut D,
    inputs: impl IntoIterator<Item = RealtimeInputFrame>,
    sampling: RealtimeSampling,
) -> Result<RealtimeTrace, Box<dyn std::error::Error + Send + Sync>>
where
    D: RealtimeEvaluationDriver,
{
    driver
        .start_trace(sampling)
        .map_err(boxed_driver_error::<D::Error>)?;
    let mut batch = None;
    let mut frames = Vec::new();
    for frame in inputs {
        match batch {
            Some(expected) if expected != frame.batch() => {
                return Err(Box::new(RealtimeTraceError::BatchChanged {
                    expected,
                    actual: frame.batch(),
                }));
            }
            None => batch = Some(frame.batch()),
            _ => {}
        }
        frames.push(
            driver
                .evaluate_frame(frame)
                .map_err(boxed_driver_error::<D::Error>)?,
        );
    }
    let generated_audio_codebooks = driver.speech_config().generated_audio_codebooks();
    driver
        .finish_trace()
        .map_err(boxed_driver_error::<D::Error>)?;
    Ok(RealtimeTrace {
        batch: batch.unwrap_or(1),
        generated_audio_codebooks,
        frames,
    })
}

fn boxed_driver_error<E>(error: E) -> Box<dyn Error + Send + Sync>
where
    E: Error + Send + Sync + 'static,
{
    Box::new(error)
}

fn transpose_frame_values<'a>(
    frames: impl IntoIterator<Item = &'a [i32]>,
    batch: usize,
    width: usize,
) -> Result<Vec<i64>, RealtimeTraceError> {
    let frames = frames.into_iter().collect::<Vec<_>>();
    for values in &frames {
        if values.len() != batch.saturating_mul(width) {
            return Err(RealtimeTraceError::FrameWidth {
                batch,
                width,
                values: values.len(),
            });
        }
    }
    let mut output = Vec::with_capacity(batch * width * frames.len());
    for batch_index in 0..batch {
        for value_index in 0..width {
            for frame in &frames {
                output.push(i64::from(frame[batch_index * width + value_index]));
            }
        }
    }
    Ok(output)
}

fn integer_tensor(
    shape: Vec<usize>,
    values: Vec<i64>,
) -> Result<TensorObservation, RealtimeTraceError> {
    Ok(TensorObservation::new(
        shape,
        TensorObservationData::I64(values),
    )?)
}

/// Invalid portable realtime trace evidence.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeTraceError {
    /// Tensor host observation failed.
    #[error(transparent)]
    Evidence(#[from] EvidenceError),
    /// Encoded input must be `[batch, codebooks, frames]`.
    #[error("encoded realtime input must have shape [batch, codebooks, frames], got {0:?}")]
    InputShape(Vec<usize>),
    /// One portable token does not fit the realtime I32 domain.
    #[error("encoded realtime token {0} does not fit I32")]
    TokenRange(i64),
    /// A frame ordinal cannot be represented in evidence.
    #[error("realtime frame index {0} does not fit I64")]
    IndexOverflow(usize),
    /// Input batch changed within one request.
    #[error("realtime trace batch changed from {expected} to {actual}")]
    BatchChanged {
        /// Initial batch.
        expected: usize,
        /// Later batch.
        actual: usize,
    },
    /// A completed frame has incompatible token geometry.
    #[error("realtime frame has {values} values for batch {batch} and width {width}")]
    FrameWidth {
        /// Trace batch.
        batch: usize,
        /// Expected values per batch row.
        width: usize,
        /// Observed values.
        values: usize,
    },
    /// Portable observation construction failed.
    #[error(transparent)]
    Observation(#[from] eredu_core::ObservationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::RealtimeFrameConvention;

    #[derive(Debug)]
    struct RecordingDriver {
        config: RealtimeSpeechConfig,
        sampling: Option<RealtimeSampling>,
        inputs: Vec<RealtimeInputFrame>,
        finishes: usize,
    }

    impl RecordingDriver {
        fn new() -> Self {
            Self {
                config: RealtimeSpeechConfig::new(
                    4,
                    2,
                    2,
                    2,
                    0,
                    0,
                    RealtimeFrameConvention::FeedbackAlignedHistory,
                    vec![0; 5],
                )
                .unwrap(),
                sampling: None,
                inputs: Vec::new(),
                finishes: 0,
            }
        }
    }

    impl RealtimeEvaluationDriver for RecordingDriver {
        type Error = std::io::Error;

        fn speech_config(&self) -> &RealtimeSpeechConfig {
            &self.config
        }

        fn start_trace(&mut self, sampling: RealtimeSampling) -> Result<(), Self::Error> {
            self.sampling = Some(sampling);
            Ok(())
        }

        fn evaluate_frame(
            &mut self,
            frame: RealtimeInputFrame,
        ) -> Result<RealtimeOutputFrame, Self::Error> {
            let value = i32::try_from(self.inputs.len()).unwrap();
            let batch = frame.batch();
            self.inputs.push(frame);
            Ok(RealtimeOutputFrame::new(
                batch,
                vec![value; batch],
                vec![value; batch * 2],
                vec![value; batch * 2],
                Some(vec![value; batch * 2]),
                Vec::new(),
            ))
        }

        fn finish_trace(&mut self) -> Result<(), Self::Error> {
            self.finishes += 1;
            Ok(())
        }
    }

    #[test]
    fn trace_runner_uses_only_the_portable_evaluation_driver() {
        let sampling = RealtimeSampling::new(0.7, 0.8, 42).unwrap();
        let mut driver = RecordingDriver::new();
        let trace = run_realtime_trace(
            &mut driver,
            [
                RealtimeInputFrame::new(1, vec![10, 11]),
                RealtimeInputFrame::new(1, vec![12, 13]),
            ],
            sampling,
        )
        .unwrap();

        assert_eq!(driver.sampling, Some(sampling));
        assert_eq!(driver.inputs.len(), 2);
        assert_eq!(driver.finishes, 1);
        assert_eq!(trace.batch(), 1);
        assert_eq!(trace.generated_audio_codebooks(), 2);
        assert_eq!(trace.frames()[1].text_tokens(), [1]);
    }

    #[test]
    fn trace_observations_transpose_frame_major_tokens() {
        let trace = RealtimeTrace {
            batch: 1,
            generated_audio_codebooks: 2,
            frames: vec![
                RealtimeOutputFrame::new(1, vec![1], vec![2, 3], vec![2, 3], None, Vec::new()),
                RealtimeOutputFrame::new(
                    1,
                    vec![4],
                    vec![5, 6],
                    vec![5, 6],
                    Some(vec![7, 8]),
                    Vec::new(),
                ),
            ],
        };
        let observations = trace.observations().unwrap();
        let Some(ObservationValue::Tensor(text)) = observations.get("trace.text_tokens") else {
            panic!("text trace must be a tensor");
        };
        assert_eq!(text.shape(), [1, 2]);
        assert_eq!(text.data(), &TensorObservationData::I64(vec![1, 4]));
    }
}
