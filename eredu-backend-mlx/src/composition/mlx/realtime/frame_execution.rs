use super::*;
use eredu_nn::RetainedTensorRoots;
use submission::{with_realtime_submission, CandidateSubmission};

/// Family-blind MLX materialization and tensor operations for prepared frames.
struct MlxRealtimeFrameTensorMechanisms<'a> {
    stream: &'a Stream,
}

impl<'a> MlxRealtimeFrameTensorMechanisms<'a> {
    /// Binds portable frame tensor operations to one selected MLX stream.
    const fn new(stream: &'a Stream) -> Self {
        Self { stream }
    }
}

impl RealtimeHostTokenMaterializer for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn materialize_i32(
        &mut self,
        values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::session_input_creation_attempt();
        let shape = shape
            .into_iter()
            .map(|dimension| {
                i32::try_from(dimension)
                    .map_err(|_| Error::Parallel("realtime tensor dimension exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Array::from_slice(values, &shape)
            .copy(self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

impl RealtimeFrameTensorMechanisms for MlxRealtimeFrameTensorMechanisms<'_> {
    type Tensor = MlxTensor;
    type Error = Error;

    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
        let batch = i32::try_from(batch)
            .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
        Array::full::<i32>(&[batch, 1], Array::from_int(token), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

struct MlxRealtimeCompletionMechanism<'a, O> {
    submission: &'a mut CandidateSubmission<O>,
}

impl<T, R, O> RealtimeCompletionMechanism<RealtimeFrameResources<'_, MlxTensor, T, R>>
    for MlxRealtimeCompletionMechanism<'_, O>
where
    T: Deref<Target = MlxKeyValueState>,
    R: RetainedTensorRoots<MlxTensor>,
    O: FnOnce() -> Result<Arc<dyn MlxRealtimeExecutionResources>, Error>,
{
    type Completion = MlxRealtimeCompletion;
    type Error = Error;

    fn begin_submission(&mut self) -> Result<(), Error> {
        self.submission.begin(|retain| retain())
    }

    fn complete(
        &mut self,
        resources: RealtimeFrameResources<'_, MlxTensor, T, R>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
    {
        let RealtimeFrameResources {
            input,
            output,
            model_state,
            payload_history,
            execution,
        } = resources;
        let mut retained = input
            .retained_payloads()
            .map(|value| value.as_array().clone())
            .collect::<Vec<_>>();
        retained.extend(
            output
                .retained_payloads()
                .map(|value| value.as_array().clone()),
        );
        retained.extend(
            output
                .diagnostics()
                .iter()
                .map(|value| value.as_array().clone()),
        );
        retained.extend(
            payload_history
                .retained_values()
                .map(|value| value.as_array().clone()),
        );
        retained.extend(model_state.retained_arrays().into_iter().cloned());
        if let Some(execution) = execution {
            execution
                .visit_retained_tensors(&mut |tensor| retained.push(tensor.as_array().clone()));
        }
        self.submission.finish(retained)
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }
}

pub(super) fn submit_scheduled_realtime_frame<P, E, R>(
    plan: RealtimeFramePlan<P>,
    executor: &mut E,
    retain_execution: impl FnOnce() -> Result<Arc<dyn MlxRealtimeExecutionResources>, Error>,
    branch: &mut MlxFrameSessionBranch,
    frame: &RealtimeInputFrame,
    stream: &Stream,
) -> Result<MlxPrepublicationFrame<P>, Error>
where
    P: RealtimeHostOutputProjection<observation::MlxHostChannels>,
    E: PreparedRealtimeFrameExecutor<
        MlxSamplingBackend,
        GenerationSampler,
        MlxKeyValueTransactionBranch,
        Error = Error,
        Retained = R,
    >,
    R: RetainedTensorRoots<MlxTensor>,
{
    let (ingress, decisions, projection) = plan.into_parts();
    let payload_contract = branch
        .payload_contract()
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let branch = branch.generation_mut();
    with_realtime_submission(
        branch,
        |branch| {
            let state = branch.model_state();
            let mut roots = state
                .model_state()
                .retained_arrays()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>();
            roots.extend(
                state
                    .payload_history()
                    .retained_values()
                    .map(|value| value.as_array().clone()),
            );
            roots.extend(
                branch
                    .random_state()
                    .map(|random| random.as_array().clone()),
            );
            roots
        },
        retain_execution,
        |branch, submission| {
            let mut host = MlxRealtimeFrameTensorMechanisms::new(stream);
            let mut tensors = MlxRealtimeFrameTensorMechanisms::new(stream);
            let mut completion = MlxRealtimeCompletionMechanism { submission };
            let submitted =
                execute_realtime_frame::<MlxSamplingBackend, _, _, _, _, _, _, _, _, _>(
                    &ingress,
                    &payload_contract,
                    frame,
                    branch,
                    &decisions,
                    &mut host,
                    &mut tensors,
                    executor,
                    &mut completion,
                    stream,
                )
                .map_err(|error| match error {
                    eredu_runtime::RealtimeFrameCoordinatorError::Completion(error)
                    | eredu_runtime::RealtimeFrameCoordinatorError::CompletionAfterSubmission(
                        error,
                    ) => error,
                    other => Error::Parallel(other.to_string()),
                })?;
            Ok(PrepublicationRealtimeFrame::new(
                submitted,
                observation::MlxRealtimeChannelObserver::new(stream),
            )
            .project_host_output(projection))
        },
    )
}
