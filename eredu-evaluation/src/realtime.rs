//! Backend-neutral execution and observation of realtime traces.

use eredu_core::{
    scheduler::{RequestId, SchedulerLimits},
    ObservationSet, ObservationValue, RealtimeBackend, RealtimeInputFrame, RealtimeModel,
    RealtimeOutputFrame, RealtimeSampling, RealtimeScheduler, TensorObservation,
    TensorObservationData,
};
use eredu_nn::Tensor;

use crate::{observe_i32_tensor, EvidenceError};

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

/// Executes portable encoded frames through any realtime backend.
pub fn run_realtime_trace<B>(
    model: &mut RealtimeModel<B>,
    inputs: impl IntoIterator<Item = RealtimeInputFrame>,
    sampling: RealtimeSampling,
) -> Result<RealtimeTrace, Box<dyn std::error::Error + Send + Sync>>
where
    B: RealtimeBackend,
{
    let request = RequestId::new(0);
    let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1)?)?;
    scheduler.register_request(model, request, sampling)?;
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
        let input = model.backend().materialize_input(model.model(), &frame)?;
        scheduler.enqueue(model, request, input)?;
        let output = loop {
            if let Some(completed) = scheduler.run_queued(model)?.pop() {
                break model.backend().observe_output(completed.output())?;
            }
            std::thread::yield_now();
        };
        frames.push(output);
    }
    scheduler.finish_request(request)?;
    Ok(RealtimeTrace {
        batch: batch.unwrap_or(1),
        generated_audio_codebooks: model.speech_config().generated_audio_codebooks(),
        frames,
    })
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
