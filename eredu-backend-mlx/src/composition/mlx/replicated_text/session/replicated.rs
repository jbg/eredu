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
    A::Unit: 'static,
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
    prepared_parameters: Vec<eredu_runtime::parameter_operations::PreparedParameterSlot>,
    parameter_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    prepared_bank_parameters: Vec<eredu_runtime::parameter_operations::PreparedBankParameter>,
    parameter_banks: std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
    >,
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
        Self::new_with_residency(prepared, store, stream, weights_stream, Default::default())
    }

    pub(super) fn new_with_residency(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
        residency: super::super::prediction::parameters::PredictionResidency,
    ) -> Result<Self, Error> {
        let mut mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream);
        mechanisms.set_prediction_residency(residency);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        let (session, facts) =
            eredu_architectures::prepared_execution::construct_selected_text_session(
                prepared, mechanisms, stream,
            )
            .map_err(Error::ArchitectureModel)?;
        let (identity, capability, model_type, residency) = facts.into_parts();
        Ok(Self::from_session(
            session, identity, capability, model_type, residency, None, None, None, true, stream,
        ))
    }
}

impl<A, S, D> CompletedReplicatedText<A, S, D>
where
    S: MlxStateMechanisms,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
    A::StaticModules: Clone,
    A::Unit: 'static,
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
            prepared_parameters: session.prepared_parameter_slots().to_vec(),
            parameter_tasks: session.parameter_materialization_tasks().to_vec(),
            prepared_bank_parameters: Vec::new(),
            session,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            parameter_banks: std::collections::BTreeMap::new(),
            #[cfg(test)]
            selected_residency,
            stream: stream.clone(),
            partition_sampling_group,
            partition_communication_authority,
            partition_sampling_rank,
            partition_public_output,
        }
    }

    pub(super) fn with_parameter_banks(
        mut self,
        parameter_banks: std::collections::BTreeMap<
            eredu_runtime::RoutedBankId,
            crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
        >,
    ) -> Result<Self, Error> {
        let mut members = Vec::new();
        for bank in parameter_banks.values() {
            members.extend(bank.prepared_parameter_members()?);
        }
        self.prepared_bank_parameters =
            eredu_runtime::parameter_operations::prepare_bank_parameter_slots(
                self.session.parameter_declarations(),
                members,
            )
            .map_err(eredu_nn::Error::backend_source)?;
        self.prepared_parameters
            .extend(self.prepared_bank_parameters.iter().map(|p| p.slot.clone()));
        self.prepared_parameters
            .sort_unstable_by(|a, b| a.parameter.id.cmp(&b.parameter.id));
        self.parameter_banks = parameter_banks;
        Ok(self)
    }

    pub(super) fn with_prediction<P>(
        mut self,
        mut prediction: SelectedPrediction<P>,
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
        let materialization = super::super::prediction::parameters::collect::<A, P>(
            &mut prediction.extension,
            &mut self.prepared_parameters,
            &mut self.parameter_tasks,
        )?;
        if materialization.transformed_weights > 0 {
            self.session
                .record_auxiliary_materialization(materialization);
        }
        Ok(CompletedReplicatedText {
            session: self.session,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            parameter_banks: self.parameter_banks,
            prepared_parameters: self.prepared_parameters,
            parameter_tasks: self.parameter_tasks,
            prepared_bank_parameters: self.prepared_bank_parameters,
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
    A::Unit: 'static,
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
    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) -> bool {
        if P::present() || !self.parameter_banks.is_empty() {
            return false;
        }
        if !self.session.visit_loaded_parameters(visitor) {
            return false;
        }
        self.prediction.visit_parameter_slots(visitor);
        true
    }

    fn parameter_materialization_tasks(
        &self,
    ) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &self.parameter_tasks
    }

    fn partition_parameter_description(
        &self,
    ) -> Option<&std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>> {
        self.session.partition_parameter_description()
    }

    fn partition_observation_hooks(
        &self,
    ) -> Option<eredu_runtime::inspection::ObservationHookSupport> {
        self.session.partition_observation_hooks()
    }

    fn prepared_parameter_slots(
        &self,
    ) -> &[eredu_runtime::parameter_operations::PreparedParameterSlot] {
        &self.prepared_parameters
    }

    fn with_parameter_slots(
        &mut self,
        location: &eredu_runtime::parameter_operations::PreparedParameterLocation,
        selected: &std::collections::BTreeSet<String>,
        operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
        stream: &Stream,
    ) -> Result<bool, Error> {
        if let eredu_runtime::parameter_operations::PreparedParameterLocation::Prediction {
            module,
        } = location
        {
            return self.prediction.with_parameter_slots(*module, operation);
        }
        if let eredu_runtime::parameter_operations::PreparedParameterLocation::Bank { bank, unit } =
            location
        {
            let owner = self
                .parameter_banks
                .values()
                .find(|owner| owner.owns_parameter_bank(*bank));
            let Some(owner) = owner else {
                return Ok(false);
            };
            return owner.with_parameter_slots(
                *bank,
                *unit,
                &self.prepared_bank_parameters,
                selected,
                operation,
                stream,
            );
        }
        self.session
            .with_parameter_slots(location, operation, stream)
            .map_err(|error| match error {
                eredu_runtime::LayerwiseAcquireError::Architecture(error) => {
                    Error::Other(Box::new(error))
                }
                eredu_runtime::LayerwiseAcquireError::Policy(error) => error,
            })
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> Result<bool, Error> {
        let banks = self.parameter_banks.values().cloned().collect::<Vec<_>>();
        let published = crate::backend::runtime::residency::parameter_bank::publish_bank_parameter_replacements(
            &banks,
            values,
            active,
            || self.session.publish_parameter_replacements(values, active),
        )?;
        if published {
            self.prediction
                .publish_parameter_replacements(values, active);
        }
        Ok(published)
    }

    fn invalidate_parameter_snapshots(&mut self) {
        self.session.invalidate_parameter_snapshots();
    }

    fn estimate_parameter_reset_state(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.session.estimate_parameter_reset_state()
    }

    fn prepare_parameter_reset_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        self.session
            .prepare_parameter_reset_state(&self.stream)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn exchange_parameter_reset_state(
        &mut self,
        slot: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        let slot = slot
            .downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("parameter reset state type differs".into()))?;
        self.session
            .exchange_parameter_reset_state(slot)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn prepare_autoregressive_cache(&mut self) -> Result<MlxPredictionTargetState, Error> {
        self.session
            .prepare_prediction_target_state(&self.stream)
            .map(MlxPredictionTargetState::new)
            .map_err(|e| Error::Other(Box::new(e)))
    }
    fn autoregressive_forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        prefill: bool,
        stream: &Stream,
    ) -> Result<Array, Error> {
        if !cache.is::<S>() {
            return Err(Error::Speculative(
                "ordinary lane state type differs".into(),
            ));
        }
        let mut lane = cache.take_state::<S>()?;
        if let Err(error) = self
            .session
            .exchange_prediction_target_state(&mut lane, stream)
        {
            cache.restore_state(lane);
            return Err(Error::Other(Box::new(error)));
        }
        let tokens = MlxTensor::from_array(tokens.clone());
        let result = self.session.sequence_logits(
            A::text_input(&tokens, None),
            if prefill {
                eredu_runtime::ExpertPass::Prefill
            } else {
                eredu_runtime::ExpertPass::Decode
            },
            stream,
        );
        let restored = match self
            .session
            .exchange_prediction_target_state(&mut lane, stream)
        {
            Ok(()) => Ok(()),
            Err(error) => super::finish_prediction_state_operation(
                Err(Error::Other(Box::new(error))),
                self.session
                    .recover_prediction_target_state_after_failure(&mut lane)
                    .map_err(|recovery| Error::Other(Box::new(recovery))),
            ),
        };
        cache.restore_state(lane);
        let output = super::finish_prediction_state_operation(
            result.map_err(|e| Error::Other(Box::new(e))),
            restored,
        )?;
        Ok(self.published(output.into_array()))
    }

    fn native_control_support(&self) -> eredu_core::execution_control::ControlSupport {
        use eredu_core::execution_control::ControlSupport;
        if D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT {
            return ControlSupport::Unsupported {
                reason: "partitioned state copies require bounded all-rank preparation agreement"
                    .into(),
            };
        }
        if P::present() {
            return ControlSupport::Unsupported {
                reason: "native text snapshots do not support selected speculative execution"
                    .into(),
            };
        }
        if self.session.estimate_control_state().is_none() {
            return ControlSupport::Unsupported {
                reason: "native state isolation or a complete storage estimate is unavailable"
                    .into(),
            };
        }
        ControlSupport::Supported
    }

    fn estimate_native_control_state(
        &self,
        saved: Option<&dyn std::any::Any>,
    ) -> Result<Option<eredu_core::execution_control::SnapshotEstimate>, Error> {
        if !matches!(
            self.native_control_support(),
            eredu_core::execution_control::ControlSupport::Supported
        ) {
            return Ok(None);
        }
        match saved {
            None => Ok(self.session.estimate_control_state()),
            Some(saved) => {
                let saved = saved.downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
                    .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
                self.session
                    .estimate_control_state_copy(saved)
                    .map_err(|error| Error::Other(Box::new(error)))
            }
        }
    }

    fn capture_native_control_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        if let eredu_core::execution_control::ControlSupport::Unsupported { reason } =
            self.native_control_support()
        {
            return Err(Error::ArchitectureModel(reason));
        }
        self.session
            .capture_control_state(&self.stream)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn estimate_native_control_growth(
        &self,
        saved: &dyn std::any::Any,
        additional: u64,
    ) -> Result<Option<u64>, Error> {
        self.validate_native_control_state(saved)?;
        let saved = saved
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .estimate_control_state_growth(saved, additional)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn copy_native_control_state(
        &mut self,
        saved: &dyn std::any::Any,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        self.validate_native_control_state(saved)?;
        let saved = saved
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .copy_control_state(saved, &self.stream)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn validate_native_control_state(&self, saved: &dyn std::any::Any) -> Result<(), Error> {
        if let eredu_core::execution_control::ControlSupport::Unsupported { reason } =
            self.native_control_support()
        {
            return Err(Error::ArchitectureModel(reason));
        }
        let saved = saved
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .validate_control_state(saved)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn exchange_native_control_state(&mut self, slot: &mut dyn std::any::Any) -> Result<(), Error> {
        self.validate_native_control_state(slot)?;
        let slot = slot
            .downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .exchange_control_state(slot, &self.stream)
            .map_err(|error| Error::Other(Box::new(error)))
    }

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
    fn speculative_activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        self.prediction.activation_execution()
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

    fn take_speculative_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        self.embedded_prediction_observers.take_activation_capture()
    }
    fn take_speculative_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        self.embedded_prediction_observers.take_activation_error()
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
            .map_err(Exception::from_source)
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
            .map_err(|error| Error::Other(Box::new(error)))?;
        let before = before_report.state_report().presence.clone();
        let before_numeric = before_report.state_report().fixed_numeric.clone();
        let before_retained = before_report.state_report().retained_numeric.clone();
        let checkpoint = self
            .session
            .checkpoint(stream)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let continuation = self
            .session
            .forward(&MlxTensor::from_array(tokens.clone()), None, stream)
            .map_err(|error| Error::Other(Box::new(error)))?
            .into_array()
            .evaluated()?
            .as_slice::<f32>()
            .to_vec();
        let advanced_report = self
            .session
            .report()
            .map_err(|error| Error::Other(Box::new(error)))?;
        let advanced = advanced_report.state_report().presence.clone();
        let advanced_numeric = advanced_report.state_report().fixed_numeric.clone();
        self.session
            .rollback(checkpoint, stream)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let restored_report = self
            .session
            .report()
            .map_err(|error| Error::Other(Box::new(error)))?;
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
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.session
            .report()
            .map(|report| report.execution_report().dense.clone())
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.session.materialization_report()
    }

    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        if self.parameter_banks.is_empty() {
            self.session.execution_strategy().parameter_bank_report()
        } else {
            self.parameter_banks.iter().map(|(id, bank)| bank.report().map(|report| (*id, report)))
                .collect::<Result<std::collections::BTreeMap<_, _>, _>>()
                .map(|reports| Some(crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport::new(reports)))
        }
    }

    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    fn reset_cache(&mut self) -> Result<(), Exception> {
        self.session
            .reset(&self.stream)
            .map_err(Exception::from_source)
    }

    fn reset_cache_distributed(&mut self) -> Result<(), Error> {
        self.session
            .reset_distributed(&self.stream)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .load_prompt_cache(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::Other(Box::new(error)))
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
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        self.session
            .load_prompt_cache_distributed(directory, expected, prefix_token_ids, &self.stream)
            .map_err(|error| Error::Other(Box::new(error)))
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
            .map_err(|error| Error::Other(Box::new(error)))
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
            .map_err(|error| Error::Other(Box::new(error)))
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
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.session
            .report()
            .map(|report| report.state_report().residency.clone())
            .map_err(Exception::from_source)
    }

    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::forward();
        let tokens = MlxTensor::from_array(tokens.clone());
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .forward_with_observer(&tokens, mask.as_ref(), stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| {
                Error::after_replicated_model_call(
                    error,
                    before,
                    self.session.successful_state_restoration_generation(),
                )
            })?;
        Ok(self.published(output))
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let tokens = input
            .and_then(|input| input::text_token_ids(input, stream).map_err(Error::from))
            .and_then(|tokens| {
                if let Some(request) = capture_geometry {
                    let shape = tokens.shape();
                    let [batch, sequence] = shape else {
                        return Err(Error::ArchitectureModel(
                            "prepared text input must have batch and sequence axes".into(),
                        ));
                    };
                    let batch =
                        u64::try_from(*batch).map_err(|error| Error::Other(Box::new(error)))?;
                    let sequence =
                        u64::try_from(*sequence).map_err(|error| Error::Other(Box::new(error)))?;
                    request
                        .validate_prefill(batch, sequence)
                        .map_err(eredu_nn::Error::backend_source)?;
                }
                Ok(tokens)
            })
            .map(MlxTensor::from_array);
        let mask = mask.cloned().map(MlxTensor::from_array);
        let input = match tokens {
            Ok(ref tokens) => Ok(A::text_input(tokens, mask.as_ref())),
            Err(error) => Err(eredu_nn::Error::backend_source(error)),
        };
        #[cfg(test)]
        if input.is_ok() {
            crate::tests::support::path_instrumentation::forward();
        }
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .prefill_input_result_with_observer(input, None, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| {
                Error::after_replicated_model_call(
                    error,
                    before,
                    self.session.successful_state_restoration_generation(),
                )
            })?;
        Ok(self.published(output))
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let tokens = tokens.map(|tokens| MlxTensor::from_array(tokens.clone()));
        let input = match tokens {
            Ok(ref tokens) => Ok(A::text_input(tokens, None)),
            Err(error) => Err(eredu_nn::Error::backend_source(error)),
        };
        #[cfg(test)]
        if input.is_ok() {
            crate::tests::support::path_instrumentation::forward();
        }
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .decode_input_result_with_observer(input, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| {
                Error::after_replicated_model_call(
                    error,
                    before,
                    self.session.successful_state_restoration_generation(),
                )
            })?;
        Ok(self.published(output))
    }
}
