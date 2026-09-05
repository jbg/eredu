use super::*;

pub(in crate::composition::mlx::replicated_text) struct CompletedReplicatedText<
    A,
    S,
    D = eredu_runtime::DirectReplicatedTextExecution,
    P = NoSelectedPrediction,
> where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    pub(super) session:
        ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    pub(super) prediction: P,
    pub(super) embedded_prediction_observers: MlxEmbeddedPredictionObservers,
    parameter_bank:
        Option<crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank>,
    #[cfg(test)]
    selected_residency: eredu_runtime::LayerWeightResidency,
    partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
    partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
    partition_sampling_rank: Option<usize>,
    partition_public_output: bool,
    pub(super) stream: Stream,
}

impl<A, S> CompletedReplicatedText<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    pub(super) fn new(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        let selected_residency = prepared.selected().residency();
        let prompt_cache_identity = prepared.prompt_cache_identity().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let mut modules = prepared.into_modules();
        let architecture = modules.take_architecture();
        let source_architecture = modules.take_source_architecture();
        let contract = modules.take_contract();
        let mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        let session = eredu_runtime::construct_replicated_text_session(
            architecture,
            source_architecture,
            contract,
            mechanisms,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(Self {
            session,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            parameter_bank: None,
            #[cfg(test)]
            selected_residency,
            partition_sampling_group: None,
            partition_communication_authority: None,
            partition_sampling_rank: None,
            partition_public_output: true,
            stream: stream.clone(),
        })
    }
}

impl<A, S, D> CompletedReplicatedText<A, S, D>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    pub(super) fn from_session(
        session: ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        prompt_cache_identity: PromptCacheModelIdentity,
        capability_estimate: eredu_architectures::capability::CapabilityEstimate,
        effective_model_type: String,
        selected_residency: eredu_runtime::LayerWeightResidency,
        partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
        partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
        partition_sampling_rank: Option<usize>,
        partition_public_output: bool,
        stream: &Stream,
    ) -> Self {
        #[cfg(not(test))]
        let _ = selected_residency;
        Self {
            session,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            parameter_bank: None,
            #[cfg(test)]
            selected_residency,
            stream: stream.clone(),
            partition_sampling_group,
            partition_communication_authority,
            partition_sampling_rank,
            partition_public_output,
        }
    }

    pub(super) fn with_parameter_bank(
        mut self,
        parameter_bank: crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
    ) -> Self {
        self.parameter_bank = Some(parameter_bank);
        self
    }

    pub(super) fn with_prediction<P>(
        self,
        prediction: SelectedPrediction<P>,
        capability: eredu_architectures::capability::CapabilityEstimate,
    ) -> Result<CompletedReplicatedText<A, S, D, SelectedPrediction<P>>, Error>
    where
        P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        >,
    {
        if prediction.extension.depth() == 0 || capability.speculative_draft_source().is_none() {
            return Err(Error::ArchitectureModel(
                "prediction extension contract is missing executable draft depth".into(),
            ));
        }
        Ok(CompletedReplicatedText {
            session: self.session,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            parameter_bank: self.parameter_bank,
            #[cfg(test)]
            selected_residency: self.selected_residency,
            partition_sampling_group: self.partition_sampling_group,
            partition_communication_authority: self.partition_communication_authority,
            partition_sampling_rank: self.partition_sampling_rank,
            partition_public_output: self.partition_public_output,
            stream: self.stream,
        })
    }
}

impl<A, S, D, P> CompletedReplicatedText<A, S, D, P>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        MlxNeuralBackend,
        S,
        MlxArchitectureLayerwisePolicy<A, S>,
        MlxArchitectureLayerwisePolicy<A, S>,
    >,
{
    fn published<T>(&self, value: T) -> T {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_publication();
        value
    }
}

impl<A, S, D, P> ErasedReplicatedTextExecutable for CompletedReplicatedText<A, S, D, P>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: ReplicatedPredictionCapability<A, S, D> + 'static,
{
    fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate {
        &self.capability_estimate
    }

    fn has_partition_control(&self) -> bool {
        self.partition_communication_authority.is_some()
    }

    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        self.partition_sampling_group.as_ref().map(|group| {
            (
                group,
                self.partition_communication_authority
                    .as_ref()
                    .expect("partition sampling group has communication authority"),
                &self.stream,
                self.partition_sampling_rank
                    .expect("partition sampling group has selected owner rank"),
            )
        })
    }

    fn partition_public_output(&self) -> bool {
        self.partition_public_output
    }

    fn with_embedded_prediction(
        &mut self,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        P::lend(self, continuation)
    }

    fn has_embedded_prediction(&self) -> bool {
        P::present()
    }

    fn install_embedded_prediction_observers(
        &mut self,
        observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        if P::present() {
            self.embedded_prediction_observers = observers;
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.selected_residency
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot {
        self.session
            .report()
            .expect("MLX state report")
            .state_report()
            .presence
            .clone()
    }

    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().fixed_numeric.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error> {
        let before_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let before = before_report.state_report().presence.clone();
        let before_numeric = before_report.state_report().fixed_numeric.clone();
        let before_retained = before_report.state_report().retained_numeric.clone();
        let checkpoint = self
            .session
            .checkpoint(stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let continuation = self
            .session
            .forward(&MlxTensor::from_array(tokens.clone()), None, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
            .into_array()
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let advanced_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let advanced = advanced_report.state_report().presence.clone();
        let advanced_numeric = advanced_report.state_report().fixed_numeric.clone();
        self.session
            .rollback(checkpoint, stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored_report = self
            .session
            .report()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let restored = restored_report.state_report().presence.clone();
        let restored_numeric = restored_report.state_report().fixed_numeric.clone();
        let restored_retained = restored_report.state_report().retained_numeric.clone();
        assert_eq!(restored_retained, before_retained);
        Ok((
            before,
            advanced,
            restored,
            before_numeric,
            advanced_numeric,
            restored_numeric,
            continuation,
        ))
    }

    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error> {
        self.session
            .report()
            .map(|report| Some(report.execution_report().residency.clone()))
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.session
            .report()
            .map(|report| report.execution_report().dense.clone())
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.session.materialization_report()
    }

    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        match &self.parameter_bank {
            Some(bank) => bank.report().map(Some),
            None => self.session.execution_strategy().parameter_bank_report(),
        }
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.session
            .reset(&self.stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        self.session
            .reset_distributed(&self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .load_prompt_cache(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .save_prompt_cache(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_distributed(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_for_input_distributed(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .save_prompt_cache_distributed(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .save_prompt_cache_for_input_distributed(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().residency.clone())
            .map_err(|error| Exception::custom(error.to_string()))
    }

    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let tokens = input::text_token_ids(input, stream)?;
        let output = self
            .session
            .prefill(&MlxTensor::from_array(tokens), None, stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let output = self
            .session
            .decode(&MlxTensor::from_array(tokens.clone()), stream)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .forward_with_observer(&tokens, mask.as_ref(), stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let tokens = input::text_token_ids(input, stream)?;
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .prefill_with_observer(&tokens, mask.as_ref(), stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }

    fn decode_with_observer(
        &mut self,
        tokens: &Array,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let tokens = MlxTensor::from_array(tokens.clone());
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let output = self
            .session
            .decode_with_observer(&tokens, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        Ok(self.published(output))
    }
}
