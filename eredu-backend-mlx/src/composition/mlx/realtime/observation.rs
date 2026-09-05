use super::*;

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

/// MLX host observer used by the neutral prepublication transition.
#[derive(Clone)]
pub struct MlxRealtimeHostObserver {
    stream: Stream,
}

impl MlxRealtimeHostObserver {
    /// Creates an observer on the selected execution stream.
    pub fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl RealtimeFrameHostObserver<MlxTensor> for MlxRealtimeHostObserver {
    type Output = RealtimeOutputFrame;
    type Error = Error;

    fn observe(
        &mut self,
        frame: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
    ) -> Result<Self::Output, Self::Error> {
        let text = frame.text().as_array();
        let batch = usize::try_from(text.dim(0))
            .map_err(|_| Error::Parallel("negative realtime output batch".into()))?;
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
        Ok(RealtimeOutputFrame::new(
            batch,
            array_i32_host(text)?,
            array_i32_host(frame.decision_audio().as_array())?,
            array_i32_host(frame.sampled_audio().as_array())?,
            frame
                .aligned_audio()
                .map(MlxTensor::as_array)
                .map(array_i32_host)
                .transpose()?,
            diagnostics,
        ))
    }
}

/// Neutral scheduler branch specialized to MLX model mechanisms.
pub type MlxFrameSessionBranch = RealtimeSessionBranch<
    RealtimePayloadBranch<MlxKeyValueTransactionBranch, MlxTensor>,
    GenerationSampler,
    RandomState,
    MlxRealtimeCompletion,
>;

/// MLX submission whose host observation must succeed before publication.
pub type MlxPrepublicationFrame =
    PrepublicationRealtimeFrame<MlxTensor, MlxRealtimeCompletion, MlxRealtimeHostObserver>;
