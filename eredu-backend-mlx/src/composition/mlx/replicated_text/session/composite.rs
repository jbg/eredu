use super::*;

pub(in crate::composition::mlx::replicated_text) trait CompositePredictionCapability<A, D>:
    Sized
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn lend(
        model: &mut CompletedComposite<A, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>>;
    fn present() -> bool;
    fn publish_parameter_replacements(
        &mut self,
        _values: &BTreeMap<String, MlxTensor>,
        _active: bool,
    ) {
    }
    fn visit_parameter_slots(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) {
    }
    fn with_parameter_slots(
        &mut self,
        _module: usize,
        _operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        None
    }
}

impl<A, D> CompositePredictionCapability<A, D> for NoSelectedPrediction
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn lend(
        _: &mut CompletedComposite<A, D, Self>,
        _: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }
    fn present() -> bool {
        false
    }
}

pub(in crate::composition::mlx::replicated_text) struct CompletedComposite<
    A,
    D = eredu_runtime::DirectReplicatedTextExecution,
    P = NoSelectedPrediction,
> where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    pub(super) session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    pub(super) admission: A::AdmissionConfig,
    pub(super) processor: eredu_runtime::SelectedProcessorExecution,
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

pub(in crate::composition::mlx::replicated_text) struct MlxCompositePredictionInput<'a, A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    pub(super) admission: &'a A::AdmissionConfig,
    pub(super) processor: &'a eredu_runtime::SelectedProcessorExecution,
}

fn validate_composite_input_selection(
    processor: &eredu_runtime::SelectedProcessorExecution,
    input: input::ModelInput<'_>,
) -> Result<(), Error> {
    if let Some(modality) = input
        .parts
        .iter()
        .map(|part| part.modality())
        .find(|modality| !processor.modalities().contains(modality))
    {
        return Err(Error::ArchitectureModel(format!(
            "prepared input modality {} is outside the selected composite modalities {:?}",
            modality.as_str(),
            processor.modalities()
        )));
    }
    if !processor.prepared_tensors()
        && input
            .parts
            .iter()
            .any(|part| part.modality() != eredu_core::InputModality::Text)
    {
        return Err(Error::ArchitectureModel(
            "prepared media tensors were not admitted by processor selection".into(),
        ));
    }
    if let Some(modality) = input.parts.iter().find_map(|part| {
        matches!(part.payload(), input::InputPayload::Embeddings(_))
            .then_some(part.modality())
            .filter(|modality| !processor.projected_modalities().contains(modality))
    }) {
        return Err(Error::ArchitectureModel(format!(
            "projected {} embeddings were not admitted by processor selection",
            modality.as_str()
        )));
    }
    Ok(())
}

pub(in crate::composition::mlx::replicated_text) fn prepare_composite_prediction_input<A>(
    admission: &A::AdmissionConfig,
    processor: &eredu_runtime::SelectedProcessorExecution,
    input: input::ModelInput<'_>,
) -> Result<
    (
        eredu_runtime::PreparedModelInput<MlxTensor>,
        eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
        Option<eredu_runtime::PreparedInputCacheIdentity>,
    ),
    Error,
>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    validate_composite_input_selection(processor, input)?;
    let supplied_cache_identity = input.cache_identity().cloned();
    let text_fingerprint = if supplied_cache_identity.is_none()
        && input.parts.iter().all(|part| {
            part.modality() == eredu_core::InputModality::Text
                && matches!(part.payload(), input::InputPayload::TokenIds(_))
        }) {
        let mut tokens = Vec::new();
        for part in input.parts {
            let input::InputPayload::TokenIds(value) = part.payload() else {
                unreachable!("text-only token payload checked above")
            };
            let value = value.evaluated()?;
            tokens.extend_from_slice(
                value
                    .try_as_slice::<u32>()
                    .map_err(|error| Error::Other(Box::new(error)))?,
            );
        }
        Some(eredu_core::cache::prompt_cache_token_fingerprint(&tokens))
    } else {
        None
    };
    let prepared = prepared_composite_input(input)?;
    if supplied_cache_identity
        .as_ref()
        .is_some_and(|identity| identity.prepared() != prepared.identity())
    {
        return Err(Error::ArchitectureModel(
            "prepared-input cache identity differs from the submitted tensors".into(),
        ));
    }
    let admitted = A::admit_prepared_input(admission, &prepared, &input::MlxTensorInputInspector)
        .map_err(|error| Error::Other(Box::new(error)))?;
    let cache_identity = match (supplied_cache_identity, text_fingerprint) {
        (Some(identity), _) => Some(identity),
        (None, Some(fingerprint)) => Some(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::Other(Box::new(error)))?,
        ),
        (None, None) => None,
    };
    Ok((prepared, admitted, cache_identity))
}

impl<A>
    eredu_architectures::speculative_execution::ReplicatedPredictionInput<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        Exception,
    > for MlxCompositePredictionInput<'_, A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Input = super::MlxModelInput;

    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        context: &Stream,
        operation: impl for<'a> FnOnce(
            <PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Input<'a>,
            MlxTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        input.with_borrowed(|input| {
            let (prepared, admitted, identity) =
                prepare_composite_prediction_input::<A>(self.admission, self.processor, input)
                    .map_err(Exception::from_source)?;
            let paired =
                PreparedCompositeInput::new(&prepared, &admitted).map_err(Exception::custom)?;
            let tokens = A::prepared_prediction_token_ids(paired, context)
                .map_err(Exception::from_source)?;
            operation(paired, tokens, identity.as_ref())
        })
    }

    fn with_decode<R>(
        &mut self,
        tokens: &MlxTensor,
        _context: &Stream,
        operation: impl for<'a> FnOnce(
            <PreparedCompositeArchitecture<A> as eredu_runtime::LayeredArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            >>::Input<'a>,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let part = input::token_ids_part(tokens.as_array()).map_err(Exception::from_source)?;
        let input = input::ModelInput::new(std::slice::from_ref(&part));
        let (prepared, admitted, _) =
            prepare_composite_prediction_input::<A>(self.admission, self.processor, input)
                .map_err(Exception::from_source)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Exception::custom)?;
        operation(paired)
    }
}

impl<A> CompletedComposite<A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
{
    pub(super) fn new(
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::new_with_residency(prepared, store, stream, weights_stream, Default::default())
    }

    pub(super) fn new_with_residency(
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
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
            eredu_architectures::prepared_execution::construct_selected_composite_session(
                prepared, mechanisms, stream,
            )
            .map_err(Error::ArchitectureModel)?;
        let (text, processor, admission) = facts.into_parts();
        let (identity, capability, model_type, residency) = text.into_parts();
        Ok(Self::from_session(
            session, admission, processor, identity, capability, model_type, residency, None, None,
            None, true, stream,
        ))
    }
}

impl<A, D, P> CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
    >,
{
    fn with_native_prediction_target_state<T>(
        &mut self,
        cache: &mut MlxPredictionTargetState,
        operation: impl FnOnce(
            &mut ReplicatedTextSession<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
            >,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if !cache.is::<MlxHybridState>() {
            return Err(Error::ArchitectureModel(
                "external-assistant target cache differs from the neutral composite state".into(),
            ));
        }
        let mut lane = cache.take_state::<MlxHybridState>()?;
        if let Err(error) = self
            .session
            .exchange_prediction_target_state(&mut lane, &self.stream)
        {
            cache.restore_state(lane);
            return Err(Error::Other(Box::new(error)));
        }
        let result = operation(&mut self.session);
        let restored = match self
            .session
            .exchange_prediction_target_state(&mut lane, &self.stream)
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
        super::finish_prediction_state_operation(result, restored)
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

    pub(super) fn with_prediction<Q>(
        mut self,
        mut prediction: SelectedPrediction<Q>,
        capability: eredu_architectures::capability::CapabilityEstimate,
    ) -> Result<CompletedComposite<A, D, SelectedPrediction<Q>>, Error>
    where
        Q: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        >,
    {
        if prediction.extension.depth() == 0 || capability.speculative_draft_source().is_none() {
            return Err(Error::ArchitectureModel(
                "prediction extension contract is missing executable draft depth".into(),
            ));
        }
        let materialization =
            super::super::prediction::parameters::collect::<PreparedCompositeArchitecture<A>, Q>(
                &mut prediction.extension,
                &mut self.prepared_parameters,
                &mut self.parameter_tasks,
            )?;
        if materialization.transformed_weights > 0 {
            self.session
                .record_auxiliary_materialization(materialization);
        }
        Ok(CompletedComposite {
            prepared_parameters: self.prepared_parameters,
            parameter_tasks: self.parameter_tasks,
            prepared_bank_parameters: self.prepared_bank_parameters,
            parameter_banks: self.parameter_banks,
            session: self.session,
            admission: self.admission,
            processor: self.processor,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            #[cfg(test)]
            selected_residency: self.selected_residency,
            partition_sampling_group: self.partition_sampling_group,
            partition_communication_authority: self.partition_communication_authority,
            partition_sampling_rank: self.partition_sampling_rank,
            partition_public_output: self.partition_public_output,
            stream: self.stream,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_session(
        session: ReplicatedTextSession<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
            D,
        >,
        admission: A::AdmissionConfig,
        processor: eredu_runtime::SelectedProcessorExecution,
        prompt_cache_identity: PromptCacheModelIdentity,
        capability_estimate: eredu_architectures::capability::CapabilityEstimate,
        effective_model_type: String,
        selected_residency: eredu_runtime::LayerWeightResidency,
        partition_sampling_group: Option<crate::backend::runtime::distributed::Group>,
        partition_communication_authority: Option<eredu_runtime::PartitionCommunicationAuthority>,
        partition_sampling_rank: Option<usize>,
        partition_public_output: bool,
        stream: &Stream,
    ) -> CompletedComposite<A, D, NoSelectedPrediction> {
        #[cfg(not(test))]
        let _ = selected_residency;
        CompletedComposite {
            prepared_parameters: session.prepared_parameter_slots().to_vec(),
            parameter_tasks: session.parameter_materialization_tasks().to_vec(),
            prepared_bank_parameters: Vec::new(),
            parameter_banks: Default::default(),
            session,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            #[cfg(test)]
            selected_residency,
            partition_sampling_group,
            partition_communication_authority,
            partition_sampling_rank,
            partition_public_output,
            stream: stream.clone(),
        }
    }

    fn prepare(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<
        (
            eredu_runtime::PreparedModelInput<MlxTensor>,
            eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
            Option<eredu_runtime::PreparedInputCacheIdentity>,
        ),
        Error,
    > {
        prepare_composite_prediction_input::<A>(&self.admission, &self.processor, input)
    }

    fn text_input(
        &self,
        tokens: &Array,
    ) -> Result<
        (
            eredu_runtime::PreparedModelInput<MlxTensor>,
            eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
            Option<eredu_runtime::PreparedInputCacheIdentity>,
        ),
        Error,
    > {
        let part = input::token_ids_part(tokens)?;
        self.prepare(input::ModelInput::new(std::slice::from_ref(&part)))
    }

    fn published<T>(&self, value: T) -> T {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::state_publication();
        value
    }
}

impl<A, D, P> ErasedReplicatedTextExecutable for CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: CompositePredictionCapability<A, D> + 'static,
{
    fn prepared_input_plans(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<
        Vec<eredu_architectures::media_plan::PreparedInputPartPlan>,
        eredu_core::CapabilityError,
    > {
        let invalid = |error: Error| eredu_core::CapabilityError::UnsupportedInput {
            architecture: self.effective_model_type.clone(),
            reason: error.to_string(),
        };
        validate_composite_input_selection(&self.processor, input).map_err(invalid)?;
        let prepared = prepared_composite_input(input).map_err(invalid)?;
        let admitted =
            A::admit_prepared_input(&self.admission, &prepared, &input::MlxTensorInputInspector)?;
        Ok(admitted.parts().iter().cloned().map(Into::into).collect())
    }

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
            .downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
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
    fn autoregressive_forward(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        prefill: bool,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let (prepared, admitted, _) = self.text_input(tokens)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let output = self.with_native_prediction_target_state(cache, |session| {
            session
                .sequence_logits(
                    paired,
                    if prefill {
                        eredu_runtime::ExpertPass::Prefill
                    } else {
                        eredu_runtime::ExpertPass::Decode
                    },
                    stream,
                )
                .map_err(|e| Error::Other(Box::new(e)))
        })?;
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
                let saved = saved.downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
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
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
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
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
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
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .validate_control_state(saved)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn exchange_native_control_state(&mut self, slot: &mut dyn std::any::Any) -> Result<(), Error> {
        self.validate_native_control_state(slot)?;
        let slot = slot
            .downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
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

    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut (dyn ErasedExternalPredictionExecutable + 'static)> {
        A::external_assistant_target_profile(&self.admission).map(|_| self as _)
    }

    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency {
        self.selected_residency
    }

    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot {
        self.session
            .report()
            .expect("MLX composite state report")
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
        let (prepared, admitted, _) = self.text_input(tokens)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let continuation = self
            .session
            .decode_input(paired, stream)
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
        self.session.execution_strategy().parameter_bank_report()
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
        let _ = (directory, expected, prefix_token_ids);
        Err(Error::ArchitectureModel(
            "composite prompt-cache loading requires the prepared-input identity".into(),
        ))
    }

    fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        self.session
            .load_prompt_cache_for_input(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
                &self.stream,
            )
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error> {
        let identity = self
            .session
            .committed_prompt_input_identity()
            .cloned()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "composite prompt cache requires a committed prepared-input identity".into(),
                )
            })?;
        self.session
            .save_prompt_cache_for_input(
                destination,
                descriptor,
                prefix_token_ids,
                options,
                &identity,
                &self.stream,
            )
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn load_prompt_cache_distributed(
        &mut self,
        _directory: &Path,
        _expected: &PromptCacheDescriptor,
        _prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error> {
        Err(Error::ArchitectureModel(
            "composite prompt-cache loading requires the prepared-input identity".into(),
        ))
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
        let identity = self
            .session
            .committed_prompt_input_identity()
            .cloned()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "composite prompt cache requires a committed prepared-input identity".into(),
                )
            })?;
        self.save_prompt_cache_for_input_distributed(
            destination,
            descriptor,
            prefix_token_ids,
            options,
            &identity,
        )
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
        if mask.is_some() {
            return Err(Error::ArchitectureModel(
                "explicit composite decoder masks require prepared-input metadata".into(),
            ));
        }
        let parts = [input::token_ids_part(tokens)?];
        self.prefill_with_observer(input::ModelInput::new(&parts), None, stream, observer)
    }

    fn prefill_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let prepared = input.and_then(|input| {
            if mask.is_some() {
                return Err(Error::ArchitectureModel(
                    "explicit composite decoder masks require prepared-input metadata".into(),
                ));
            }
            let prepared = self.prepare(input)?;
            if let Some(request) = capture_geometry {
                let [batch, sequence] = prepared.1.decoder_shape();
                request
                    .validate_prefill(batch, sequence)
                    .map_err(eredu_nn::Error::backend_source)?;
            }
            Ok(prepared)
        });
        let (paired, cache_identity) = match prepared {
            Ok((ref prepared, ref admitted, ref identity)) => (
                PreparedCompositeInput::new(prepared, admitted).map_err(eredu_nn::Error::backend),
                identity.clone(),
            ),
            Err(error) => (Err(eredu_nn::Error::backend_source(error)), None),
        };
        #[cfg(test)]
        if paired.is_ok() {
            crate::tests::support::path_instrumentation::forward();
        }
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .prefill_input_result_with_observer(paired, cache_identity, stream, &mut observer)
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
        let prepared = tokens.and_then(|tokens| self.text_input(tokens));
        let paired = match prepared {
            Ok((ref prepared, ref admitted, _)) => {
                PreparedCompositeInput::new(prepared, admitted).map_err(eredu_nn::Error::backend)
            }
            Err(error) => Err(eredu_nn::Error::backend_source(error)),
        };
        #[cfg(test)]
        if paired.is_ok() {
            crate::tests::support::path_instrumentation::forward();
        }
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .decode_input_result_with_observer(paired, stream, &mut observer)
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

impl<A, D, P> ErasedExternalPredictionExecutable for CompletedComposite<A, D, P>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
    P: 'static,
{
    fn prepare_external_prediction_target_cache(
        &mut self,
    ) -> Result<MlxPredictionTargetState, Error> {
        self.session
            .prepare_prediction_target_state(&self.stream)
            .map(MlxPredictionTargetState::new)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn prefill_external_prediction_target(
        &mut self,
        input: input::ModelInput<'_>,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error> {
        let paths = A::external_prediction_capture_paths(request)
            .map_err(|error| Error::Other(Box::new(error)))?
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "assistant capture request differs from the neutral target architecture".into(),
                )
            })?;
        let mut observer = ExactPredictionCaptureObserver::new(paths)?;
        let (prepared, admitted, _) = self.prepare(input)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let request = request.clone();
        let captured = observer.values.clone();
        let stream = self.stream.clone();
        self.with_native_prediction_target_state(cache, |session| {
            session
                .prefill_input_with_capture(paired, &stream, &mut observer, |forward| {
                    let values = captured
                        .borrow()
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| {
                            value.ok_or_else(|| {
                                eredu_nn::Error::backend(format!(
                                    "external-assistant target did not reach capture path {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    A::external_prediction_capture(&request, forward, values)?.ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "architecture did not produce its selected assistant capture",
                        )
                    })
                })
                .map_err(|error| Error::Other(Box::new(error)))
        })
    }

    fn verify_external_prediction_target(
        &mut self,
        tokens: &MlxTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error> {
        let (prepared, admitted, _) = self.text_input(tokens.as_array())?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
        let paths = A::external_prediction_capture_paths(request)
            .map_err(|error| Error::Other(Box::new(error)))?
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "assistant capture request differs from the neutral target architecture".into(),
                )
            })?;
        let mut observer = ExactPredictionCaptureObserver::new(paths)?;
        let request = request.clone();
        let captured = observer.values.clone();
        let stream = self.stream.clone();
        self.with_native_prediction_target_state(cache, |session| {
            session
                .decode_input_with_capture(paired, &stream, &mut observer, |forward| {
                    let values = captured
                        .borrow()
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(index, value)| {
                            value.ok_or_else(|| {
                                eredu_nn::Error::backend(format!(
                                    "external-assistant target did not reach capture path {index}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    A::external_prediction_capture(&request, forward, values)?.ok_or_else(|| {
                        eredu_nn::Error::backend(
                            "architecture did not produce its selected assistant capture",
                        )
                    })
                })
                .map_err(|error| Error::Other(Box::new(error)))
        })
    }

    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, MlxTensor>,
    ) -> Result<MlxTensor, Error> {
        self.session
            .apply_prediction_target_operation(
                CompositePredictionTargetOperation { operation },
                &self.stream,
            )
            .map_err(|error| Error::Other(Box::new(error)))
    }
}

impl<A, S, D, P> ReplicatedPredictionCapability<A, S, D> for SelectedPrediction<P>
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
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, MlxTensor>,
        active: bool,
    ) {
        super::super::prediction::parameters::publish::<A, P>(&mut self.extension, values, active);
    }
    fn visit_parameter_slots(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) {
        super::super::prediction::parameters::visit::<A, P>(&mut self.extension, visitor);
    }
    fn with_parameter_slots(
        &mut self,
        module: usize,
        operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
    ) -> Result<bool, Error> {
        super::super::prediction::parameters::with_slots::<A, P>(
            &mut self.extension,
            module,
            operation,
        )
    }
    fn activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        self.extension.activation_execution(&self.selected)
    }

    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let mut strategy =
            eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy::<
                A,
                MlxNeuralBackend,
                S,
                MlxReplicatedTextMechanisms<A, S>,
                D,
                P,
                super::prepared_speculative::MlxTextPredictionInput,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
            >::new(
                &mut model.session,
                &mut model.prediction.extension,
                selected,
                super::prepared_speculative::MlxTextPredictionInput,
                &model.stream,
            );
        let observers = std::mem::take(&mut model.embedded_prediction_observers);
        let mut executor = eredu_architectures::speculative_execution::EmbeddedPredictionExecutor::<
            _,
            super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
        >::with_observers(&mut strategy, observers);
        let result = {
            let mut erased = eredu_architectures::speculative_execution::DynEmbeddedExecutor::<
                super::prepared_speculative::MlxEmbeddedExecutorTypes,
            >::new(&mut executor);
            continuation.execute(selected, &mut erased)
        };
        model.embedded_prediction_observers = executor.into_observers();
        Some(result)
    }

    fn present() -> bool {
        true
    }
}
