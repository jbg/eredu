//! Native observation of completed categorical and dense realtime channels.

use eredu_core::RealtimeDecisionDiagnostics;
use eredu_runtime::{CompletedRealtimeFrame, RealtimeFrameHostObserver, RealtimeHostChannelValues};
use safemlx::{Array, Dtype, Stream};

use crate::{backend::error::Error, MlxTensor};

fn array_i32_host(array: &Array) -> Result<Vec<i32>, Error> {
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated.as_slice::<i32>().to_vec()),
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Int64 => evaluated
            .as_slice::<i64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        dtype => Err(Error::Parallel(format!(
            "realtime token observation expected integer values, got {dtype:?}"
        ))),
    }
}

fn array_f32_host(array: &Array, stream: &Stream) -> Result<Vec<f32>, Error> {
    let array = if array.dtype() == Dtype::Float32 {
        array.clone()
    } else {
        array.as_dtype(Dtype::Float32, stream)?
    };
    Ok(array.evaluated()?.as_slice::<f32>().to_vec())
}

/// Converts a completed native channel frame to its portable host representation.
///
/// Channel declarations determine tensor geometry and categorical or dense
/// conversion. Architecture-specific output interpretation follows this step.
#[derive(Clone)]
pub struct MlxRealtimeChannelObserver {
    stream: Stream,
}

impl MlxRealtimeChannelObserver {
    /// Creates an observer on the selected execution stream.
    pub fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl RealtimeFrameHostObserver<CompletedRealtimeFrame<MlxTensor, MlxTensor>>
    for MlxRealtimeChannelObserver
{
    type Output = CompletedRealtimeFrame<RealtimeHostChannelValues, RealtimeDecisionDiagnostics>;
    type Error = Error;

    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    ) -> Result<Self::Output, Self::Error> {
        let values = observe_channels(frame, frame.values(), &self.stream)?;
        let aligned = observe_channels(frame, frame.aligned(), &self.stream)?;
        let diagnostics = frame
            .diagnostics()
            .iter()
            .enumerate()
            .map(|(prediction, logits)| {
                let logits = logits.as_array();
                let shape = logits
                    .shape()
                    .iter()
                    .map(|dimension| {
                        usize::try_from(*dimension).map_err(|_| {
                            Error::Parallel("negative realtime diagnostic dimension".into())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                RealtimeDecisionDiagnostics::new(
                    prediction,
                    shape,
                    array_f32_host(logits, &self.stream)?,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        CompletedRealtimeFrame::new(
            frame.channels().clone(),
            frame.batch(),
            values,
            aligned,
            frame.decision_channels().to_vec(),
            diagnostics,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

fn observe_channels(
    frame: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    channels: &[eredu_runtime::RealtimeFrameChannel<MlxTensor>],
    stream: &Stream,
) -> Result<Vec<eredu_runtime::RealtimeFrameChannel<eredu_runtime::RealtimeHostChannelValues>>, Error>
{
    channels
        .iter()
        .map(|value| {
            let declaration = frame
                .channels()
                .channel(value.channel())
                .ok_or_else(|| Error::Parallel("undeclared completed realtime channel".into()))?;
            let shape = match &declaration.payload {
                eredu_core::RealtimeChannelPayload::Categorical { width, .. } => {
                    vec![frame.batch(), *width]
                }
                eredu_core::RealtimeChannelPayload::Dense { shape } => {
                    std::iter::once(frame.batch())
                        .chain(shape.iter().copied())
                        .collect()
                }
            };
            let actual = value
                .payload()
                .as_array()
                .shape()
                .iter()
                .map(|&dimension| {
                    usize::try_from(dimension)
                        .map_err(|_| Error::Parallel("negative realtime channel dimension".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if actual != shape {
                return Err(Error::Parallel(format!(
                    "realtime channel {:?} has shape {actual:?}, expected {shape:?}",
                    value.channel()
                )));
            }
            let payload = match &declaration.payload {
                eredu_core::RealtimeChannelPayload::Categorical { .. } => {
                    eredu_runtime::RealtimeHostChannelValues::Categorical(array_i32_host(
                        value.payload().as_array(),
                    )?)
                }
                eredu_core::RealtimeChannelPayload::Dense { .. } => {
                    if !matches!(
                        value.payload().as_array().dtype(),
                        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64
                    ) {
                        return Err(Error::Parallel(
                            "realtime dense channel requires floating-point values".into(),
                        ));
                    }
                    eredu_runtime::RealtimeHostChannelValues::Dense(array_f32_host(
                        value.payload().as_array(),
                        stream,
                    )?)
                }
            };
            Ok(eredu_runtime::RealtimeFrameChannel::new(
                value.channel(),
                value.coordinate(),
                payload,
            ))
        })
        .collect()
}
