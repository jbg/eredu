use super::*;

/// MLX-owned prefill input.
///
/// Arrays are cloned handles, not copied tensor storage. Owning the handles
/// makes submission independent of the caller's temporary `ModelInput` view.
#[derive(Debug, Clone)]
pub struct MlxModelInput {
    parts: Vec<input::InputPart>,
    cache_identity: Option<eredu_runtime::PreparedInputCacheIdentity>,
}

impl From<input::ModelInput<'_>> for MlxModelInput {
    fn from(input: input::ModelInput<'_>) -> Self {
        Self {
            parts: input.parts.to_vec(),
            cache_identity: input.cache_identity().cloned(),
        }
    }
}

impl MlxModelInput {
    /// Converts processor-owned MLX values into an opaque backend prompt.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub fn from_prepared(input: &PreparedModelInput) -> Self {
        let mut owned = input.with_model_input(|borrowed| Self::from(borrowed));
        owned.cache_identity = input.cache_identity().cloned();
        owned
    }

    /// Returns the exact semantic identity carried by processor-produced input.
    pub const fn cache_identity(&self) -> Option<&eredu_runtime::PreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Couples manually prepared tensors to caller-owned semantic content.
    pub fn with_semantic_content_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
    ) -> Result<Self, Error> {
        let prepared = eredu_runtime::PreparedModelInput::new(self.parts.clone(), |array| {
            eredu_runtime::PreparedInputInspector::identity(&input::MlxInputInspector, array)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.cache_identity = Some(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?,
        );
        Ok(self)
    }

    /// Borrows the owned input parts as a model-input view for one operation.
    pub fn with_borrowed<T>(&self, execute: impl FnOnce(input::ModelInput<'_>) -> T) -> T {
        let input = match self.cache_identity.as_ref() {
            Some(identity) => input::ModelInput::with_cache_identity(&self.parts, identity),
            None => input::ModelInput::new(&self.parts),
        };
        execute(input)
    }
}

/// The single MLX implementation of architecture-erased prefill and decode.
///
/// Cache state and optional communication belong to the same selected model
/// session so callers cannot accidentally execute a sharded model with an
/// unrelated communicator.
pub struct MlxModelSession {
    model: Executable,
    submission_in_flight: Rc<Cell<Option<u64>>>,
    next_submission_ticket: Cell<u64>,
    floating_state_dtype_bytes: std::num::NonZeroU8,
    distributed: Option<MlxDistributedSession>,
    capabilities: eredu_core::SessionCapabilities,
    state_residency: CacheResidencyPolicy,
    #[cfg(any(feature = "image", feature = "audio"))]
    processor: Option<ModelProcessor>,
}

impl MlxModelSession {
    /// Creates a session and validates that its communicator matches the model topology.
    pub(crate) fn from_model(
        mut model: MlxModel,
        admitted_capabilities: eredu_core::SessionCapabilities,
    ) -> Result<Self, Error> {
        let floating_state_dtype_bytes = model.floating_state_dtype_bytes();
        let state_residency = model.state_residency().clone();
        #[cfg(any(feature = "image", feature = "audio"))]
        let processor = model.take_processor();
        let distributed = model.take_distributed();
        let mut executable = model.into_executable();
        executable
            .reset_cache_with_options(state_residency.clone())
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let realized_capabilities = eredu_core::SessionCapabilities::new(true, true, true);
        if admitted_capabilities != realized_capabilities {
            return Err(Error::ArchitectureModel(format!(
                "realized MLX session capabilities {realized_capabilities:?} do not match pre-materialization admission {admitted_capabilities:?}"
            )));
        }
        Ok(Self {
            model: executable,
            submission_in_flight: Rc::new(Cell::new(None)),
            next_submission_ticket: Cell::new(1),
            floating_state_dtype_bytes,
            distributed,
            capabilities: realized_capabilities,
            state_residency,
            #[cfg(any(feature = "image", feature = "audio"))]
            processor,
        })
    }

    pub(in crate::composition::mlx) const fn floating_state_dtype_bytes(
        &self,
    ) -> std::num::NonZeroU8 {
        self.floating_state_dtype_bytes
    }

    fn begin_submission(&self) -> Result<SessionSubmissionLease, Error> {
        if self.submission_in_flight.get().is_some() {
            return Err(Error::ArchitectureModel(
                "model session already owns an unresolved submission completion".into(),
            ));
        }
        let ticket = self.next_submission_ticket.get();
        let next = ticket.checked_add(1).ok_or_else(|| {
            Error::ArchitectureModel("model session submission ticket space exhausted".into())
        })?;
        self.next_submission_ticket.set(next);
        self.submission_in_flight.set(Some(ticket));
        Ok(SessionSubmissionLease {
            owner: self.submission_in_flight.clone(),
            ticket,
        })
    }

    fn ensure_no_submission_in_flight(&self) -> Result<(), Error> {
        if self.submission_in_flight.get().is_some() {
            Err(Error::ArchitectureModel(
                "model session state cannot be changed while a submission completion is unresolved"
                    .into(),
            ))
        } else {
            Ok(())
        }
    }

    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn processor(&self) -> Option<&ModelProcessor> {
        self.processor.as_ref()
    }

    pub(crate) fn effective_model_type(&self) -> &str {
        self.model.effective_model_type()
    }

    /// Reports how the session-owned model exposes speculative weights.
    pub fn speculative_capability(&self) -> SpeculativeCapability {
        self.model.speculative_capability()
    }

    /// Installs causal observers on this session's selected embedded-prediction executor.
    pub fn install_embedded_prediction_observers<TensorObserver, LogitsObserver>(
        &mut self,
        tensors: TensorObserver,
        logits: LogitsObserver,
    ) -> Result<(), Error>
    where
        TensorObserver: RuntimeActivationObserver<MlxTensor, Exception> + 'static,
        LogitsObserver: RuntimeActivationObserver<Array, Exception> + 'static,
    {
        self.ensure_no_submission_in_flight()?;
        let observers =
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::new(
                tensors, logits,
            );
        if self.model.install_embedded_prediction_observers(observers) {
            Ok(())
        } else {
            Err(Error::ArchitectureModel(
                "session has no selected embedded-prediction executor".into(),
            ))
        }
    }

    /// Returns bounded parameter-residency telemetry when available.
    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.model.residency_report()
    }

    /// Returns dense checkpoint-streaming telemetry when enabled.
    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.model.dense_stream_report()
    }

    /// Returns sparse routed-expert cache telemetry when enabled.
    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        self.model.parameter_bank_report()
    }

    /// Returns the complete model-derived identity for a reusable prompt cache.
    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Error> {
        self.model.prompt_cache_model_identity().map_err(Into::into)
    }

    pub(in crate::composition::mlx) fn capability_estimate(
        &self,
    ) -> Result<eredu_architectures::capability::CapabilityEstimate, eredu_core::CapabilityError>
    {
        self.model.architecture_capability_estimate()
    }

    pub(in crate::composition::mlx) fn prepared_input_part_plan(
        &self,
        input: &crate::backend::runtime::media::input::InputPart,
    ) -> Result<eredu_architectures::media_plan::PreparedInputPartPlan, eredu_core::CapabilityError>
    {
        self.model.prepared_input_part_plan(input)
    }

    pub(in crate::composition::mlx) fn speculative_model_mut(&mut self) -> &mut Executable {
        &mut self.model
    }

    #[cfg(test)]
    pub(crate) fn neutral_prediction_target_mut(
        &mut self,
    ) -> Result<&mut dyn super::super::replicated_text::ErasedReplicatedTextExecutable, Error> {
        Ok(self.model.erased_mut())
    }

    /// Clears all MLX cache state under the authoritative selected policy.
    pub fn reset(&mut self) -> Result<(), Error> {
        self.ensure_no_submission_in_flight()?;
        if self.model.has_neutral_partitioned_control() {
            return self.model.reset_cache_distributed().map_err(Into::into);
        }
        self.model
            .reset_cache_with_options(self.state_residency.clone())
            .map_err(Into::into)
    }

    /// Submits one cached decode position from a portable token id.
    pub fn submit_token_decode(
        &mut self,
        backend: &MlxBackend<'_>,
        token_id: u32,
    ) -> Result<Submission<MlxModelOutput, MlxSessionCompletion>, Error> {
        let token_ids = [token_id];
        let input =
            Array::from(token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        self.decode(backend, input)
    }

    /// Returns aggregate cache-residency telemetry for this session.
    pub fn cache_residency_report(
        &self,
    ) -> Result<Option<eredu_runtime::CacheResidencyReport>, Error> {
        self.model
            .cache_residency_report()
            .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Atomically persists the completed prefix owned by this session.
    pub fn save_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        if self.model.has_neutral_partitioned_control() {
            self.model
                .save_prompt_cache_distributed(root, descriptor, prefix_token_ids, options)?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                })
        } else {
            self.model
                .save_prompt_cache(
                    root,
                    descriptor,
                    prefix_token_ids,
                    options,
                    backend.stream(),
                )
                .map_err(Into::into)
        }
    }

    /// Opens a compatible persisted prefix and replaces this session's cache.
    pub fn load_prompt_cache(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        self.ensure_no_submission_in_flight()?;
        let root = root.as_ref();
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let options = options.clone();
        let manifest = if self.model.has_neutral_partitioned_control() {
            self.model
                .load_prompt_cache_distributed(root, expected, prefix_token_ids)?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                })?
        } else {
            self.model.load_prompt_cache(
                root,
                expected,
                prefix_token_ids,
                options,
                backend.stream(),
            )?
        };
        Ok(manifest)
    }

    /// Opens a persisted prefix only when it matches an exact prepared input.
    pub fn load_prompt_cache_for_input(
        &mut self,
        backend: &MlxBackend<'_>,
        root: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input: &MlxModelInput,
    ) -> Result<PromptCacheManifest, Error> {
        self.ensure_no_submission_in_flight()?;
        let identity = input.cache_identity.clone().ok_or_else(|| {
            Error::ArchitectureModel(
                "prompt-cache loading requires prepared-input semantic identity".into(),
            )
        })?;
        let CacheResidencyPolicy::Paged(options) = &self.state_residency else {
            return Err(Error::ArchitectureModel(
                "prompt-cache loading requires paged state selected during preparation".into(),
            ));
        };
        let root = root.as_ref();
        if self.model.has_neutral_partitioned_control() {
            self.model
                .load_prompt_cache_for_input_distributed(
                    root,
                    expected,
                    prefix_token_ids,
                    identity,
                )?
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "this partition rank owns no prompt-cache state".into(),
                    )
                })
        } else {
            self.model
                .load_prompt_cache_for_input(
                    root,
                    expected,
                    prefix_token_ids,
                    identity,
                    options.clone(),
                    backend.stream(),
                )
                .map_err(Into::into)
        }
    }

    /// Returns communication when this is a distributed session.
    pub const fn distributed(&self) -> Option<&MlxDistributedSession> {
        self.distributed.as_ref()
    }

    pub(super) fn synchronizes_sampling(&self) -> bool {
        self.model.erased().partition_sampling_context().is_some()
    }

    /// Samples on the canonical rank and synchronizes the result for this
    /// distributed model session.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_synchronize<S: Sampler<MlxSamplingBackend>>(
        &self,
        logits: Option<&MlxTensor>,
        batch_size: i32,
        sampler: &mut S,
        temperature: f32,
        prng_state: Option<&mut RandomState>,
        finished: bool,
    ) -> Result<crate::backend::runtime::distributed::parallel::SynchronizedToken, Error> {
        let executable = self.model.erased();
        let (group, authority, stream, sampling_rank) =
            executable.partition_sampling_context().ok_or_else(|| {
                Error::Parallel(
                    "sampling synchronization requires a distributed model session".into(),
                )
            })?;
        crate::backend::runtime::distributed::parallel::sample_and_synchronize_bounded(
            logits,
            batch_size,
            sampler,
            temperature,
            prng_state,
            finished,
            sampling_rank,
            group,
            authority,
            stream,
        )
    }

    /// Submits instrumented prefill through this selected MLX session.
    pub fn submit_prefill_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: MlxModelInput,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let submission_lease = self.begin_submission()?;
        let result = (|| {
            let token_validation_scope = TokenValidationScope::begin()?;
            let output = input.with_borrowed(|input| {
                self.model.erased_mut().prefill_with_observer(
                    input,
                    None,
                    backend.stream(),
                    &mut ArrayObserverAdapter { inner: observer },
                )
            })?;
            Ok(model_array_submission(
                output,
                token_validation_scope.finish(),
                submission_lease.clone(),
            ))
        })();
        if result.is_err() {
            submission_lease.release();
        }
        result
    }

    fn submit_decode_with_observer(
        &mut self,
        backend: &MlxBackend<'_>,
        input: Array,
        observer: &mut impl RuntimeActivationObserver<MlxTensor, Exception>,
    ) -> Result<Submission<Array, MlxSessionCompletion>, Error> {
        let submission_lease = self.begin_submission()?;
        let result = (|| {
            let token_validation_scope = TokenValidationScope::begin()?;
            let output = self.model.erased_mut().decode_with_observer(
                &input,
                backend.stream(),
                &mut ArrayObserverAdapter { inner: observer },
            )?;
            Ok(model_array_submission(
                output,
                token_validation_scope.finish(),
                submission_lease.clone(),
            ))
        })();
        if result.is_err() {
            submission_lease.release();
        }
        result
    }
}

impl<'a> BackendSession<MlxBackend<'a>> for MlxModelSession {
    type PrefillInput = MlxModelInput;
    type DecodeInput = Array;
    type Output = MlxModelOutput;
    type Completion = MlxSessionCompletion;

    fn capabilities(&self) -> eredu_core::SessionCapabilities {
        self.capabilities
    }

    fn prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        let submission_lease = self.begin_submission()?;
        let result = (|| {
            let token_validation_scope = TokenValidationScope::begin()?;
            let output = input
                .with_borrowed(|input| prefill_model(&mut self.model, input, backend.stream()))?;
            let public_output = self.model.erased().partition_public_output();
            Ok(model_submission(
                output,
                token_validation_scope.finish(),
                public_output,
                submission_lease.clone(),
            ))
        })();
        if result.is_err() {
            submission_lease.release();
        }
        result
    }

    fn decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
    ) -> Result<Submission<Self::Output, Self::Completion>, Error> {
        let submission_lease = self.begin_submission()?;
        let result = (|| {
            let token_validation_scope = TokenValidationScope::begin()?;
            let output = decode_model(&mut self.model, &input, backend.stream())?;
            let public_output = self.model.erased().partition_public_output();
            Ok(model_submission(
                output,
                token_validation_scope.finish(),
                public_output,
                submission_lease.clone(),
            ))
        })();
        if result.is_err() {
            submission_lease.release();
        }
        result
    }

    fn observe_output(
        &self,
        backend: &MlxBackend<'a>,
        output: &Self::Output,
    ) -> Result<ObservationSet, Error> {
        let mut observations = ObservationSet::new();
        if let Some(logits) = output.logits() {
            observations
                .insert(
                    eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                    ObservationValue::Tensor(observe_tensor(logits, backend.stream())?),
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        }
        Ok(observations)
    }
}

impl<'a> InspectableBackendSession<MlxBackend<'a>> for MlxModelSession {
    fn inspect_prefill(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::PrefillInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let submission = self.submit_prefill_with_observer(backend, input, &mut collector)?;
        let logits = submission.wait()?;
        let output = if self.model.erased().partition_public_output() {
            MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
        } else {
            MlxModelOutput::new(None)
        };
        let observations = collector.materialize(backend.stream())?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }

    fn inspect_decode(
        &mut self,
        backend: &MlxBackend<'a>,
        input: Self::DecodeInput,
        request: &ObservationRequest,
    ) -> Result<InspectedOutput<Self::Output>, Error> {
        let mut collector = InspectionCollector::new(request);
        let submission = self.submit_decode_with_observer(backend, input, &mut collector)?;
        let logits = submission.wait()?;
        let output = if self.model.erased().partition_public_output() {
            MlxModelOutput::new(Some(MlxTensor::from_array(logits)))
        } else {
            MlxModelOutput::new(None)
        };
        let observations = collector.materialize(backend.stream())?;
        Ok(InspectedOutput {
            output,
            observations,
        })
    }
}

impl<'a> TextGenerationBackend for MlxBackend<'a> {
    type Prompt = MlxModelInput;
    type Token = MlxTextToken;
    type TextGenerationState = MlxTextGenerationState;
    type TextCompletion = MlxTextCompletion;

    fn start_text_generation(
        _: &Self,
        config: TextGenerationConfig,
    ) -> Result<Self::TextGenerationState, Error> {
        let sampling = config.sampling();
        let prng = if sampling.temperature == 0.0 {
            None
        } else {
            Some(RandomState::from_key(safemlx::random::key(config.seed())?))
        };
        let sampler = match config.strategy() {
            TextSamplingStrategy::Standard => {
                MlxTextSampler::Standard(GenerationSampler::from_resolved(sampling))
            }
            TextSamplingStrategy::MirostatV2 { tau, eta } => {
                let sampler = MirostatV2Sampler::new(tau, eta)
                    .map_err(|error| eredu_core::BackendError::Execution {
                        session: "text-generation".into(),
                        operation: "configure Mirostat V2".into(),
                        message: error.to_string(),
                    })?
                    .penalties(
                        sampling.repetition_penalty,
                        sampling.repeat_last_n,
                        sampling.frequency_penalty,
                        sampling.presence_penalty,
                    );
                MlxTextSampler::MirostatV2(sampler)
            }
        };
        Ok(MlxTextGenerationState {
            temperature: sampling.temperature,
            prng,
            sampler,
        })
    }

    fn prepare_text_prompt(
        backend: &Self,
        prompt_token_ids: Vec<u32>,
    ) -> Result<Self::Prompt, Error> {
        if prompt_token_ids.is_empty() {
            return Err(Error::ArchitectureModel(
                "text generation requires at least one prompt token".into(),
            ));
        }
        let tokens =
            Array::from(prompt_token_ids.as_slice()).try_index_device(NewAxis, backend.stream())?;
        let parts = [input::input_part(
            InputModality::Text,
            input::InputPayload::TokenIds(tokens),
            [],
            [],
        )?];
        MlxModelInput::from(input::ModelInput::new(&parts)).with_semantic_content_fingerprint(
            eredu_core::cache::prompt_cache_token_fingerprint(&prompt_token_ids),
        )
    }

    fn submit_text_prefill(
        runtime: &mut ModelRuntime<Self>,
        prompt: Self::Prompt,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let submission = runtime.prefill(prompt)?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
    }

    fn submit_text_decode(
        runtime: &mut ModelRuntime<Self>,
        token: Self::Token,
        filter: &TokenFilter,
        state: &mut Self::TextGenerationState,
    ) -> Result<Submission<Self::Token, Self::TextCompletion>, Error> {
        let stream = runtime.backend().stream().clone();
        let input = token.value.try_index_device((.., NewAxis), &stream)?;
        let submission = runtime.decode(input)?;
        sample_text_submission(runtime.session(), submission, filter, state, stream)
    }
}

fn model_array_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    submission_lease: SessionSubmissionLease,
) -> Submission<Array, MlxSessionCompletion> {
    let mut retained = Vec::with_capacity(1 + token_validations.arrays().count());
    retained.push(output.clone());
    retained.extend(token_validations.arrays().cloned());
    Submission {
        output,
        completion: MlxSessionCompletion {
            inner: MlxSessionCompletionKind::Model {
                token_validations,
                _retained: retained,
                submission_lease,
            },
        },
    }
}

pub(super) fn model_submission(
    output: Array,
    token_validations: TokenValidationBatch,
    public_output: bool,
    submission_lease: SessionSubmissionLease,
) -> Submission<MlxModelOutput, MlxSessionCompletion> {
    let submission = model_array_submission(output, token_validations, submission_lease);
    Submission {
        output: MlxModelOutput::new(
            public_output.then(|| MlxTensor::from_array(submission.output)),
        ),
        completion: submission.completion,
    }
}
