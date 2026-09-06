use super::*;

/// Family-blind MLX materialization and tensor operations for prepared frames.
pub struct MlxRealtimeFrameTensorMechanisms<'a> {
    stream: &'a Stream,
}

impl<'a> MlxRealtimeFrameTensorMechanisms<'a> {
    /// Binds portable frame tensor operations to one selected MLX stream.
    pub const fn new(stream: &'a Stream) -> Self {
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

    fn column(
        &mut self,
        matrix: &Self::Tensor,
        column: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        let column = i32::try_from(column)
            .map_err(|_| Error::Parallel("realtime tensor column exceeds i32".into()))?;
        matrix
            .as_array()
            .try_index_device((.., column..column + 1), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn filled_column(&mut self, token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
        let batch = i32::try_from(batch)
            .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
        Array::full::<i32>(&[batch, 1], Array::from_int(token), self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }

    fn stack_columns(
        &mut self,
        columns: &[Self::Tensor],
        batch: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        if columns.is_empty() {
            let batch = i32::try_from(batch)
                .map_err(|_| Error::Parallel("realtime tensor batch exceeds i32".into()))?;
            return Array::zeros::<i32>(&[batch, 0], self.stream)
                .map(MlxTensor::from_array)
                .map_err(Into::into);
        }
        let columns = columns.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        stack_axis(&columns, 1, self.stream)?
            .squeeze_axes(&[-1], self.stream)
            .map(MlxTensor::from_array)
            .map_err(Into::into)
    }
}

/// Exact MLX completion creation for one shared-coordinator Moshi frame.
pub struct MlxRealtimeFrameCompletionMechanism {
    token_validations: Option<TokenValidationScope>,
    execution_resources: Option<Arc<neutral_moshi::SelectedRealtimeResources>>,
}

impl MlxRealtimeFrameCompletionMechanism {
    /// Starts the one token-validation scope owned by this frame submission.
    pub fn begin() -> Result<Self, Error> {
        TokenValidationScope::begin()
            .map(|token_validations| Self {
                token_validations: Some(token_validations),
                execution_resources: None,
            })
            .map_err(Into::into)
    }

    fn begin_for_execution(
        resources: Arc<neutral_moshi::SelectedRealtimeResources>,
    ) -> Result<Self, Error> {
        Self::begin().map(|mut mechanism| {
            mechanism.execution_resources = Some(resources);
            mechanism
        })
    }
}

impl<T>
    RealtimeFrameCompletionMechanism<
        MlxTensor,
        T,
        (
            MlxTensor,
            eredu_architectures::moshi::ForwardContext<MlxTensor>,
        ),
    > for MlxRealtimeFrameCompletionMechanism
where
    T: Deref<Target = MlxKeyValueState>,
{
    type Completion = MlxRealtimeCompletion;
    type Error = Error;

    fn complete(
        &mut self,
        input: MaterializedRealtimeInput<MlxTensor>,
        output: &CompletedRealtimeFrame<MlxTensor, MlxTensor>,
        model_state: &T,
        payload_history: &RealtimePayloadHistory<MlxTensor>,
        execution: Option<(
            MlxTensor,
            eredu_architectures::moshi::ForwardContext<MlxTensor>,
        )>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
    {
        let token_validations = self.token_validations.take().ok_or_else(|| {
            RealtimeCompletionCreationError::before_submission(Error::Parallel(
                "MLX realtime completion scope was already consumed".into(),
            ))
        })?;
        let (_, _, input_audio, forced_audio, forced_text, _, _) = input.into_parts();
        let mut retained = vec![input_audio.into_array()];
        retained.extend(forced_audio.map(MlxTensor::into_array));
        retained.extend(forced_text.map(MlxTensor::into_array));
        retained.extend([
            output.text().as_array().clone(),
            output.decision_audio().as_array().clone(),
            output.sampled_audio().as_array().clone(),
        ]);
        retained.extend(output.aligned_audio().map(|value| value.as_array().clone()));
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
        if let Some((text_logits, forward)) = execution {
            retained.push(text_logits.into_array());
            retained.extend(
                forward
                    .temporal_mask()
                    .map(|value| value.as_array().clone()),
            );
            retained.extend(
                forward
                    .temporal_output()
                    .map(|value| value.as_array().clone()),
            );
            retained.extend(forward.text_logits().map(|value| value.as_array().clone()));
            retained.extend(
                forward
                    .previous_depth_token()
                    .map(|value| value.as_array().clone()),
            );
        }
        // Failed event construction can still own accepted work. Its completion
        // crosses the error boundary with the resources until terminal proof.
        MlxRealtimeCompletion::submit_retained_with_resources(
            retained,
            token_validations.finish(),
            self.execution_resources.take(),
        )
    }

    fn retained_resources(&self, completion: &Self::Completion) -> usize {
        completion.retained_resources()
    }
}

struct MlxSelectedRealtimeFrameExecutor<'a> {
    model: &'a mut MlxRealtimeExecution,
}

impl
    PreparedRealtimeFrameExecutor<
        MlxSamplingBackend,
        GenerationSampler,
        MlxKeyValueTransactionBranch,
    > for MlxSelectedRealtimeFrameExecutor<'_>
{
    type Error = Error;
    type Retained = (
        MlxTensor,
        eredu_architectures::moshi::ForwardContext<MlxTensor>,
    );

    fn execute(
        &mut self,
        model_state: &mut MlxKeyValueTransactionBranch,
        temporal: &[MlxTensor],
        driver: &mut eredu_runtime::SequentialDecisionDriver<MlxSamplingBackend, GenerationSampler>,
        context: &Stream,
    ) -> Result<Self::Retained, Self::Error> {
        self.model
            .execute_selected_realtime(&mut *model_state, temporal, driver, context)
    }
}

pub(super) fn submit_scheduled_realtime_frame(
    model: &mut MoshiRealtimeExecution<MlxRealtimeExecution>,
    branch: &mut MlxFrameSessionBranch,
    frame: &RealtimeInputFrame,
    stream: &Stream,
) -> Result<MlxPrepublicationFrame, Error> {
    model.executor().validate_stream(stream)?;
    let ingress = eredu_architectures::moshi::realtime_ingress_contract(model.execution_config())
        .map_err(Error::ArchitectureModel)?;
    // Reject invalid host tokens before entering the mutable native operation.
    // This reuses the neutral contract; invalid user input does not poison a
    // model whose execution has not begun.
    ingress
        .validate(frame)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    let payload_contract = branch
        .payload_contract(&ingress)
        .map_err(|error| Error::Parallel(error.to_string()))?;
    model.executor_mut().with_submission(|model| {
        let mut host = MlxRealtimeFrameTensorMechanisms::new(stream);
        let mut tensors = MlxRealtimeFrameTensorMechanisms::new(stream);
        let mut completion =
            MlxRealtimeFrameCompletionMechanism::begin_for_execution(model.completion_resources())?;
        let mut executor = MlxSelectedRealtimeFrameExecutor { model };
        let submitted = execute_realtime_frame::<MlxSamplingBackend, _, _, _, _, _, _, _, _>(
            &ingress,
            &payload_contract,
            frame,
            branch.generation_mut(),
            &eredu_architectures::moshi::realtime_decision_execution(),
            &mut host,
            &mut tensors,
            &mut executor,
            &mut completion,
            stream,
        )
        .map_err(|error| Error::Parallel(error.to_string()))?;
        Ok(PrepublicationRealtimeFrame::new(
            submitted,
            MlxRealtimeHostObserver::new(stream),
        ))
    })
}
