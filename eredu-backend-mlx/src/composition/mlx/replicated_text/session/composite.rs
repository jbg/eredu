use super::*;
mod decode_input;
mod external_prefill;
mod external_state;

mod media_prefill;
use media_prefill::{
    NativeMediaCaptureValidation, NativeMediaPrefillEntry, NativeOriginalMediaBinding,
    NativeOriginalMediaPrefillEntry,
};

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
    fn prepare_original_prediction(
        _model: &CompletedComposite<A, D, Self>,
        _context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        None
    }
    fn present() -> bool;
    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        <Self as CompositePredictionCapability<A, D>>::collect_retained_storage(
            self,
            &mut storage,
        )?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;

    fn count_parameter_owners(
        &self,
        _counts: &mut ParameterOwnerCounts,
        _guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        Err(ParameterOwnerSourceError::PredictionUnavailable)
    }

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
    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        <Self as CompositePredictionCapability<A, D>>::collect_retained_storage(
            self,
            &mut storage,
        )?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn count_parameter_owners(
        &self,
        _counts: &mut ParameterOwnerCounts,
        _guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        Ok(())
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
    media_prefill: Option<NativeMediaPrefillEntry<A, D>>,
    bind_original_media: Option<NativeOriginalMediaBinding<A>>,
    original_media_prefill: Option<NativeOriginalMediaPrefillEntry<A, D>>,
    media_capture_validation: Option<NativeMediaCaptureValidation<A>>,
    pub(super) admission: A::AdmissionConfig,
    pub(super) processor: eredu_runtime::SelectedProcessorExecution,
    prompt_cache_identity: PromptCacheModelIdentity,
    capability_estimate: eredu_architectures::capability::CapabilityEstimate,
    effective_model_type: String,
    pub(super) prediction: P,
    pub(super) embedded_prediction_observers: MlxEmbeddedPredictionObservers,
    // Only the built-in no-op observers certify no retained payload. An erased
    // installed observer requires its own inventory before model promotion.
    prediction_observers_are_empty: bool,
    prepared_parameters: Vec<eredu_runtime::parameter_operations::PreparedParameterSlot>,
    parameter_tasks: Vec<eredu_runtime::ReplicatedTextMaterializationTask>,
    prepared_bank_parameters: Vec<eredu_runtime::parameter_operations::PreparedBankParameter>,
    parameter_banks: std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        crate::backend::runtime::residency::parameter_bank::IndexedBankSource,
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
    for part in input.parts {
        eredu_architectures::media_plan::validate_selected_part(
            processor,
            part.modality(),
            part.payload().kind(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
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
        Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ),
    Error,
>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    validate_composite_input_selection(processor, input)?;
    let supplied_cache_identity = input.shared_cache_identity().cloned().or_else(|| {
        // Legacy raw inputs need one owner. Managed input already carries its
        // exact shared source and must never rebuild this payload here.
        input
            .cache_identity()
            .cloned()
            .map(eredu_runtime::SharedPreparedInputCacheIdentity::new)
    });
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
        (None, Some(fingerprint)) => Some(eredu_runtime::SharedPreparedInputCacheIdentity::new(
            prepared
                .cache_identity(fingerprint)
                .map_err(|error| Error::Other(Box::new(error)))?,
        )),
        (None, None) => None,
    };
    Ok((prepared, admitted, cache_identity))
}

impl<A>
    eredu_architectures::speculative_execution::ReplicatedPredictionInput<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
        Error,
    > for MlxCompositePredictionInput<'_, A>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Input = super::MlxModelInput;

    type Prefill = eredu_architectures::speculative_execution::AdmittedPredictionPrefill<
        A::AdmissionConfig,
        MlxTensor,
        input::MlxTensorInputInspector,
        A::InputPartPlan,
    >;

    fn with_prefill_source<R>(
        &mut self,
        input: Self::Input,
        _context: &Stream,
        operation: impl FnOnce(Result<Self::Prefill, Error>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let chunk = input.prefill_chunk_positions();
            let prepared =
                prepare_composite_prediction_input::<A>(self.admission, self.processor, input).map(
                    |(prepared, admitted, identity)| {
                        eredu_architectures::speculative_execution::AdmittedPredictionPrefill::new(
                            prepared,
                            admitted,
                            self.admission.clone(),
                            input::MlxTensorInputInspector,
                            identity,
                            chunk,
                        )
                    },
                );
            operation(prepared)
        })
    }

    fn with_prefill_source_with_metadata<R>(
        &mut self,
        input: Self::Input,
        prepared: eredu_runtime::input::PreparedModelInputOwner<MlxTensor>,
        _context: &Stream,
        metadata: &eredu_nn::workspace::WorkspaceContext,
        operation: impl FnOnce(Result<Self::Prefill, Error>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let source=(|| {
                metadata.charge_metadata(std::mem::size_of::<(
                    Self::Prefill,Result<Self::Prefill,Error>,A::AdmissionConfig,
                    eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
                    input::MlxTensorInputInspector,
                    Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
                )>()).map_err(|cause|Error::Neural(cause.into()))?;
                for part in prepared.parts() {
                    eredu_architectures::media_plan::validate_selected_part(self.processor,part.modality(),part.payload().kind())
                        .map_err(|cause|Error::Neural(metadata.metadata_source(cause)))?;
                }
                let config=A::retain_admission_config_with_metadata(self.admission,metadata)?;
                let inspector=input::MlxTensorInputInspector;
                let admitted=A::admit_prepared_input_with_metadata(&config,&prepared,&inspector,metadata)?;
                eredu_architectures::speculative_execution::AdmittedPredictionPrefill::from_prepared_owner_with_metadata(
                    prepared,admitted,config,inspector,input.shared_cache_identity().cloned(),input.prefill_chunk_positions(),metadata,
                ).map_err(Error::Neural)
            })();
            operation(source)
        })
    }

    fn requested_chunks(input: &Self::Input) -> Option<std::num::NonZeroU64> {
        input.with_borrowed(|input| input.prefill_chunk_positions())
    }

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
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        input.with_borrowed(|input| {
            let (prepared, admitted, identity) =
                prepare_composite_prediction_input::<A>(self.admission, self.processor, input)?;
            let paired = PreparedCompositeInput::new(&prepared, &admitted)
                .map_err(Error::ArchitectureModel)?;
            let tokens = A::prepared_prediction_token_ids(paired, context).map_err(Error::from)?;
            operation(paired, tokens, identity.as_ref().map(AsRef::as_ref))
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
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let part = input::token_ids_part(tokens.as_array())?;
        let input = input::ModelInput::new(std::slice::from_ref(&part));
        let (prepared, admitted, _) =
            prepare_composite_prediction_input::<A>(self.admission, self.processor, input)?;
        let paired =
            PreparedCompositeInput::new(&prepared, &admitted).map_err(Error::ArchitectureModel)?;
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
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::new_with_residency(prepared, store, stream, weights_stream, Default::default())
    }

    pub(super) fn new_with_residency(
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        stream: &Stream,
        weights_stream: &Stream,
        residency: super::super::prediction::parameters::PredictionResidency,
    ) -> Result<Self, Error> {
        Self::new_with_prepared_layerwise(prepared, store, stream, weights_stream, residency, None)
    }

    pub(super) fn new_with_prepared_layerwise(
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        stream: &Stream,
        weights_stream: &Stream,
        residency: super::super::prediction::parameters::PredictionResidency,
        layerwise_manager: Option<
            crate::backend::runtime::execution::generic::PreparedLayerwiseManager,
        >,
    ) -> Result<Self, Error> {
        let mut mechanisms = MlxReplicatedTextMechanisms::new(store, stream, weights_stream)?;
        mechanisms.set_prediction_residency(residency);
        mechanisms.set_prepared_layerwise_manager(layerwise_manager);
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
        external_state::with_state::<A,D,T>(
            &mut self.session, cache, &self.stream, None, |session, _| operation(session),
        )
    }

    pub(super) fn with_parameter_banks(
        mut self,
        parameter_banks: std::collections::BTreeMap<
            eredu_runtime::RoutedBankId,
            crate::backend::runtime::residency::parameter_bank::IndexedBankSource,
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
            media_prefill: self.media_prefill,
            bind_original_media: self.bind_original_media,
            original_media_prefill: self.original_media_prefill,
            media_capture_validation: self.media_capture_validation,
            admission: self.admission,
            processor: self.processor,
            prompt_cache_identity: self.prompt_cache_identity,
            capability_estimate: capability,
            effective_model_type: self.effective_model_type,
            prediction,
            embedded_prediction_observers: self.embedded_prediction_observers,
            prediction_observers_are_empty: self.prediction_observers_are_empty,
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
            media_prefill: None,
            bind_original_media: None,
            original_media_prefill: None,
            media_capture_validation: None,
            admission,
            processor,
            prompt_cache_identity,
            capability_estimate,
            effective_model_type,
            prediction: NoSelectedPrediction,
            embedded_prediction_observers: MlxEmbeddedPredictionObservers::default(),
            prediction_observers_are_empty: true,
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
            Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        ),
        Error,
    > {
        #[cfg(test)]
        input::record_original_semantic_preparation();
        prepare_composite_prediction_input::<A>(&self.admission, &self.processor, input)
    }

    fn text_input(
        &self,
        tokens: &Array,
    ) -> Result<
        (
            eredu_runtime::PreparedModelInput<MlxTensor>,
            eredu_architectures::media_plan::AdmittedCompositeInput<A::InputPartPlan>,
            Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        ),
        Error,
    > {
        decode_input::prepare::<A>(&self.admission, &self.processor, tokens, None)
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
    fn current_media_semantic_binding(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::MediaSessionBinding,
        eredu_runtime::replicated_session::MediaSemanticBindingError<Error>,
    > {
        self.session.media_semantic_binding()
    }
    fn prepare_original_media_semantic_binding(
        &self,
        funding: &eredu_nn::workspace::WorkspaceMetadataFunding,
    ) -> Result<(), eredu_runtime::replicated_session::OriginalMediaBindingError> {
        self.session.prepare_original_media_semantic_binding(funding).map(|_| ())
    }

    #[cfg(test)]
    fn prepare_completed_media_binding_fixture(
        &self,
    ) -> Result<(), eredu_runtime::replicated_session::MediaSemanticBindingError<Error>> {
        // Explicit ordinary setup after model loading; never called by B bind.
        self.session
            .prepare_ordinary_media_semantic_binding()
            .map(|_| ())
    }
    fn original_request_media_binding(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::MediaSessionBinding,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        self.session.original_request_media_binding()
    }

    fn bind_completed_original_media_semantics(
        &self,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    ) -> Result<
        eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        super::super::CompletedMediaBindingError,
    > {
        use super::super::CompletedMediaBindingError;
        let Some(bind) = self.bind_original_media else {
            return Err(CompletedMediaBindingError::boundary(
                original,
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ));
        };
        let binding = match self.session.media_semantic_binding() {
            Ok(v) => v,
            Err(cause) => return Err(CompletedMediaBindingError::mechanism(original, cause)),
        };
        bind(&self.admission, original, blueprint, source, binding)
            .map_err(CompletedMediaBindingError::semantic)
    }

    fn bind_original_media_semantics(
        &self,
        original: eredu_architectures::media_plan::OriginalPreparedMediaSemantics<'_>,
        blueprint: &eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
    ) -> Result<
        eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        eredu_core::BackendFailure,
    > {
        let Some(bind) = self.bind_original_media else {
            return Err(eredu_core::BackendFailure::from_error(
                original.reject_boundary(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ),
            ));
        };
        let binding = match self.session.prepare_ordinary_media_semantic_binding() {
            Ok(binding) => binding,
            Err(cause) => {
                return Err(eredu_core::BackendFailure::from_error(
                    media_prefill::OriginalBindingFailure {
                        cause: Error::Other(Box::new(cause)),
                        original: original.reject_boundary(
                            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                        ),
                    },
                ));
            }
        };
        bind(&self.admission, original, blueprint, source, binding)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn inference_execution_identity(
        &self,
    ) -> &eredu_runtime::working_memory::InferenceExecutionIdentity {
        self.session.inference_execution_identity()
    }

    fn indexed_bank_sources(&self)->Option<&std::collections::BTreeMap<eredu_runtime::RoutedBankId,crate::backend::runtime::residency::parameter_bank::IndexedBankSource>>{
        Some(&self.parameter_banks)
    }

    fn install_workspace_parameter_representations(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        let controls = [
            std::mem::size_of::<(&Self, &eredu_nn::workspace::WorkspaceContext)>(),
            std::mem::size_of::<eredu_runtime::replicated_session::RuntimeInspectionBoundary>(),
            std::mem::size_of::<
                Result<
                    Result<(), eredu_nn::Error>,
                    eredu_runtime::replicated_session::RuntimeInspectionBoundary,
                >,
            >(),
        ];
        context.charge_metadata(
            controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        self.session
            .inspect_runtime_execution_fixed(|_, _, execution| {
                super::super::partitioned::parameter_representation::install_with_static_fallback(
                    context,
                    D::static_modules_ref(execution),
                    |visitor| D::visit_parameter_sources(execution, visitor, context),
                )
            })
            .map_err(|cause| context.metadata_source(cause))?
    }

    fn count_parameter_owners(
        &self,
        guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<ParameterOwnerCounts, ParameterOwnerSourceError> {
        self.session
            .inspect_runtime_execution_fixed(|_, _, execution| {
                let static_modules = D::static_modules_ref(execution)
                    .ok_or(ParameterOwnerSourceError::StaticUnavailable)?;
                let mut counts = ParameterOwnerCounts::default();
                counts.observe_source(ParameterOwnerRole::Static, None, static_modules, guard)?;
                self.prediction.count_parameter_owners(&mut counts, guard)?;
                Ok(counts)
            })
            .map_err(ParameterOwnerSourceError::Boundary)?
    }

    fn collect_retained_target_module_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        storage.include_retained_values(|visitor| {
            self.session
                .visit_retained_values(visitor)
                .map_err(|error| Error::Other(Box::new(error)))
        })
    }

    fn collect_retained_prediction_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        // Establish the same idle-session fence before borrowing extension owners.
        self.session
            .inspect_runtime(|_, _| Ok(()))
            .map_err(|error| Error::Other(Box::new(error)))?;
        self.prediction.collect_retained_storage(storage)
    }

    fn collect_retained_target_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_target_module_storage(storage)?;
        self.session
            .inspect_runtime(|mechanisms, state| {
                mechanisms.collect_retained_storage(state, storage)
            })
            .map_err(|error| Error::Other(Box::new(error)))?;
        for bank in self.parameter_banks.values() {
            bank.collect_retained_storage(storage)?;
        }
        Ok(())
    }

    fn collect_retained_target_nonstate_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_target_module_storage(storage)?;
        self.session
            .inspect_runtime(|mechanisms, _| mechanisms.collect_retained_parameter_storage(storage))
            .map_err(|error| Error::Other(Box::new(error)))?;
        for bank in self.parameter_banks.values() {
            bank.collect_retained_storage(storage)?;
        }
        Ok(())
    }

    fn collect_retained_decoder_state_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.session
            .inspect_runtime_state(|state| {
                state.collect_retained_storage(storage).map_err(Error::from)
            })
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn retained_inference_authority(
        &self,
    ) -> Result<eredu_runtime::working_memory::InferenceRetention, Error> {
        self.session
            .inspect_runtime_state(|state| {
                Ok(
                    eredu_runtime::working_memory::InferenceStateRetention::inference_retention(
                        state,
                    )
                    .clone(),
                )
            })
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn original_state_slot_facts(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::cache::state::NativeStateSlotCounts>,
        eredu_runtime::replicated_session::RuntimeInspectionBoundary,
    > {
        let result = self
            .session
            .inspect_runtime_execution_fixed(|_, state, _| {
                Ok::<_, std::convert::Infallible>(state.retained_owner_slot_counts())
            })?;
        match result {
            Ok(value) => Ok(value),
            Err(never) => match never {},
        }
    }

    #[cfg(test)]
    fn inspection_adapters_for_test(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::inspection_adapters_for_test(&self.session, geometry)
    }
    #[cfg(test)]
    fn prefill_status_for_test(&self) -> Result<(bool, bool), Error> {
        MlxReplicatedTextMechanisms::prefill_status_for_test(&self.session)
    }

    fn bind_layerwise_neural_recipe(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::bind_session_neural_recipe(&self.session, pool, recipe, funding)
    }

    fn prefill_control_facts(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        geometry: eredu_core::InferenceGeometry,
        graph_capacity: std::num::NonZeroU64,
        native_root_capacity: Option<u64>,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<Option<eredu_runtime::working_memory::TextPrefillScopeFacts>, Error> {
        MlxReplicatedTextMechanisms::session_prefill_control_facts(
            &self.session,
            pool,
            geometry,
            graph_capacity,
            native_root_capacity,
            retained_sources,
            native_recipe,
            funding,
        )
    }
    fn prepare_original_operation_banks(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        original: &eredu_runtime::working_memory::OriginalTextPrefillScopeSet,
        step: &eredu_runtime::working_memory::InferenceTextStep,
        registration: crate::backend::runtime::execution::generic::OriginalOperationRegistration,
        controls: eredu_runtime::working_memory::OriginalTextControlGuard,
        host_destinations: Option<eredu_runtime::working_memory::OriginalHostDestinationBank>,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        funding: Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Result<
        Option<crate::backend::runtime::execution::generic::OriginalOperationBankOwner>,
        Error,
    > {
        MlxReplicatedTextMechanisms::prepare_session_original_operations(
            &self.session,
            pool,
            original,
            step,
            registration,
            controls,
            host_destinations,
            retained_sources,
            native_recipe,
            funding,
        )
    }
    fn native_storage_mechanism(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>,
        Error,
    > {
        MlxReplicatedTextMechanisms::session_native_storage(&self.session).map(Some)
    }
    fn prefill_roots_runtime(&self) -> Result<safemlx::PrefillRootsRuntime, Error> {
        MlxReplicatedTextMechanisms::session_prefill_roots_runtime(&self.session)
    }
    fn install_parallel_control(
        &self,control:Option<crate::backend::runtime::distributed::topology::original_source::control::OriginalParallelControlProjection>,
    )->Result<(),Error>{
        MlxReplicatedTextMechanisms::install_session_parallel_control(&self.session,control)
    }
    fn install_prefill_controls(
        &self,
        controls: Option<crate::backend::submission_recovery::prefill::PrefillControlProjection>,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::install_session_prefill_controls(&self.session, controls)
    }
    fn prepare_opening_rows(
        &self,
        selection: eredu_runtime::layered::BoundCaptureSelection<'_>,
    ) -> Result<super::super::NativeOpeningRowsPlan, Error> {
        MlxReplicatedTextMechanisms::prepare_session_opening_rows(&self.session, selection)
    }
    fn install_opening_rows(
        &self,
        rows: &super::super::NativeOpeningRowsOwner,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::install_session_opening_rows(&self.session, rows)
    }

    fn retire_expired_opening_rows(&self) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::retire_session_opening_rows(&self.session)?;
        MlxReplicatedTextMechanisms::retire_session_prefill_controls(&self.session)
    }

    #[cfg(test)]
    fn opening_rows_status_for_test(&self) -> Result<(bool, bool), Error> {
        MlxReplicatedTextMechanisms::opening_rows_status_for_test(&self.session)
    }

    #[cfg(test)]
    fn busy_rows_retirement_for_test(&self) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::busy_rows_retirement_for_test(&self.session)
    }

    fn validate_media_capture_input(
        &self,
        input: input::ModelInput<'_>,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<(), Error> {
        let validate = self.media_capture_validation.ok_or_else(|| {
            Error::Other(Box::new(
                eredu_core::PreparedControlInputError::InstrumentationUnavailable,
            ))
        })?;
        validate(&self.admission, self.prepare(input), geometry)
    }

    fn shared_observation_paths(&self) -> Option<&eredu_runtime::SharedLayeredObservationPaths> {
        self.session.shared_observation_paths()
    }

    fn validate_prepared_observation_paths(
        &self,
        expected: &eredu_runtime::SharedLayeredObservationPaths,
    ) -> Result<(), Error> {
        self.session
            .validate_prepared_observation_paths(expected)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn collect_retained_idle_auxiliary_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.session
            .inspect_runtime(|_, _| Ok(()))
            .map_err(|error| Error::Other(Box::new(error)))?;
        self.session
            .inspect_runtime_state(|state| state.collect_retained_host_storage(storage))
            .map_err(|error| Error::Other(Box::new(error)))?;
        super::super::state::collect_partition_auxiliary_storage(
            storage,self.partition_sampling_group.as_ref(),self.partition_communication_authority.as_ref(),
        )?;
        collect_snapshot_shared_sources(
            storage,
            self.session.shared_observation_paths(),
            self.session.committed_shared_prompt_input_identity(),
            self.prediction_observers_are_empty,
        )
        .map_err(Error::from)
    }

    fn collect_snapshot_storage_fixed(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), crate::backend::runtime::residency::storage::SnapshotStorageInspectionError>
    {
        self.session
            .inspect_runtime_execution_fixed(|_, state, _| {
                state.collect_retained_storage_fixed(storage)?;
                state.collect_retained_host_storage_fixed(storage)
            })??;
        super::super::state::collect_partition_auxiliary_storage(
            storage,self.partition_sampling_group.as_ref(),self.partition_communication_authority.as_ref(),
        )?;
        collect_snapshot_shared_sources(
            storage,
            self.session.shared_observation_paths(),
            self.session.committed_shared_prompt_input_identity(),
            self.prediction_observers_are_empty,
        )?;
        Ok(())
    }

    fn layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::MetalAllocationFacts,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        self.session
            .inspect_runtime(|mechanisms, _| mechanisms.layerwise_workspace(allocation))
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn project_resident_workspace_with_storage(
        &self,
        batch: std::num::NonZeroU32,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::nn::workspace::ProjectedResidentState, Error> {
        self.session
            .inspect_runtime_state(|state| {
                state
                    .project_resident_workspace_with_storage(batch, context)
                    .map_err(Error::from)
            })
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn original_text_frontier(
        &self,
    ) -> Result<Option<u64>, super::super::prediction::OriginalTextFrontierError> {
        use super::super::prediction::OriginalTextFrontierError;
        use eredu_runtime::ReplicatedTextSessionMechanisms;
        self.session
            .inspect_runtime_execution_fixed(|mechanisms, state, _| {
                mechanisms.original_prefill_state_frontier(state)
            })
            .map_err(OriginalTextFrontierError::Boundary)?
            .map_err(OriginalTextFrontierError::Mechanism)
    }

    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error> {
        self.session
            .inspect_runtime_state(|state| state.validate_text_frontier(expected))
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn prepared_control_binding(
        &self,
    ) -> Result<super::super::prediction::PreparedControlBinding, Error> {
        use eredu_runtime::ReplicatedTextSessionMechanisms;
        use eredu_runtime::working_memory::InferenceStateRetention;
        let origin = self
            .session
            .control_state_origin()
            .map_err(|e| Error::Other(Box::new(e)))?;
        self.session
            .inspect_runtime(|mechanisms, state| {
                let frontier = mechanisms.prefill_state_frontier(state)?.ok_or_else(|| {
                    Error::Other(Box::new(
                        eredu_core::PreparedControlInputError::UnknownFrontier,
                    ))
                })?;
                Ok(super::super::prediction::PreparedControlBinding {
                    origin,
                    revision: state.inference_retention().revision().clone(),
                    frontier,
                })
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }

    fn validate_prepared_control_binding(
        &self,
        binding: &super::super::prediction::PreparedControlBinding,
    ) -> Result<(), Error> {
        use eredu_runtime::ReplicatedTextSessionMechanisms;
        use eredu_runtime::working_memory::InferenceStateRetention;
        self.session
            .validate_control_state_origin(&binding.origin)
            .map_err(|e| Error::Other(Box::new(e)))?;
        self.session
            .inspect_runtime(|mechanisms, state| {
                state
                    .inference_retention()
                    .validate_revision(&binding.revision)
                    .map_err(|e| Error::Other(Box::new(e)))?;
                if mechanisms.prefill_state_frontier(state)? != Some(binding.frontier) {
                    return Err(Error::Other(Box::new(
                        eredu_core::PreparedControlInputError::SourceMismatch,
                    )));
                }
                Ok(())
            })
            .map_err(|e| Error::Other(Box::new(e)))
    }

    fn prepared_control_attribution(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<eredu_core::PreparedPromptAttribution, Error> {
        use eredu_core::{
            PreparedControlInputError as E, PreparedPromptSegment, PromptTokenAttribution,
        };
        let binding = self.prepared_control_binding()?;
        validate_composite_input_selection(&self.processor, input)?;
        let prepared = prepared_composite_input(input)?;
        if input
            .cache_identity()
            .is_some_and(|identity| identity.prepared() != prepared.identity())
        {
            return Err(Error::Other(Box::new(E::SourceMismatch)));
        }
        let admitted =
            A::admit_prepared_input(&self.admission, &prepared, &input::MlxTensorInputInspector)
                .map_err(|e| Error::Other(Box::new(e)))?;
        let mut plans = Vec::with_capacity(admitted.parts().len());
        admitted
            .visit_prompt_segments(|plan| plans.push(plan))
            .map_err(|e| Error::Other(Box::new(e)))?;
        let mut canonical_token_ids = Vec::new();
        let mut segments = Vec::with_capacity(plans.len());
        for (plan, part) in plans.into_iter().zip(input.parts) {
            let tokens = match part.payload() {
                input::InputPayload::TokenIds(value) => {
                    let start = canonical_token_ids.len() as u64;
                    let value = value.evaluated()?;
                    match value.as_array().dtype() {
                        Dtype::Uint32 => canonical_token_ids.extend_from_slice(
                            value
                                .try_as_slice::<u32>()
                                .map_err(|e| Error::Other(Box::new(e)))?,
                        ),
                        Dtype::Int32 => {
                            for &token in value
                                .try_as_slice::<i32>()
                                .map_err(|e| Error::Other(Box::new(e)))?
                            {
                                canonical_token_ids.push(
                                    u32::try_from(token).map_err(|e| Error::Other(Box::new(e)))?,
                                );
                            }
                        }
                        _ => return Err(Error::Other(Box::new(E::InvalidAttribution))),
                    }
                    PromptTokenAttribution::Canonical {
                        range: [start, canonical_token_ids.len() as u64],
                    }
                }
                _ => PromptTokenAttribution::NotTokenized,
            };
            segments.push(PreparedPromptSegment { plan, tokens });
        }
        let semantic_content_identity = match input.cache_identity() {
            Some(identity) => identity.semantic_content_fingerprint().to_owned(),
            None if segments
                .iter()
                .all(|s| matches!(s.tokens, PromptTokenAttribution::Canonical { .. })) =>
            {
                eredu_core::cache::prompt_cache_token_fingerprint(&canonical_token_ids)
            }
            None => return Err(Error::Other(Box::new(E::MissingSourceIdentity))),
        };
        let result = eredu_core::PreparedPromptAttribution {
            schema_version: eredu_core::PREPARED_PROMPT_ATTRIBUTION_VERSION,
            prepared: prepared.identity().clone(),
            semantic_content_identity,
            opening_position: binding.frontier,
            decoder_positions: admitted.decoder_positions(),
            batch: 1,
            segments,
            canonical_token_ids,
        };
        result.validate().map_err(|e| Error::Other(Box::new(e)))?;
        Ok(result)
    }

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
        let banks = self.parameter_banks.values().map(|source|source.storage().clone()).collect::<Vec<_>>();
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
    fn autoregressive_prefill(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut MlxPredictionTargetState,
        sample: bool,
        cancellation: &eredu_core::GenerationCancellationToken,
        stream: &Stream,
    ) -> Result<
        eredu_core::SpeculativePrefillOutcome<
            eredu_runtime::speculative::autoregressive::AutoregressivePrefill<Array>,
        >,
        Error,
    > {
        use eredu_core::SpeculativePrefillOutcome as P;
        use eredu_runtime::replicated_session::PrefillSourceOutcome as O;
        let chunk = input.prefill_chunk_positions();
        let prepared = self.prepare(input);
        let shape = prepared
            .as_ref()
            .ok()
            .map(|(_, admitted, _)| admitted.decoder_shape());
        let admission = self.admission.clone();
        let (outcome, completed) = self.with_native_prediction_target_state(cache, |session| {
            let before = session.successful_state_restoration_generation();
            let make_source = |geometry| {
                let Ok((input, _, _)) = &prepared else {
                    return Ok(None);
                };
                // Source identity belongs to this lane, not the canonical
                // session whose selected execution is borrowed for the pass.
                eredu_architectures::prefill::PreparedCompositeTextPrefill::from_prepared_text(
                    input,
                    geometry,
                    admission.clone(),
                    input::MlxTensorInputInspector,
                )
            };
            let result = if sample {
                session
                    .try_prefill_unbudgeted_source_progress_cancellable(
                        shape,
                        chunk,
                        make_source,
                        cancellation,
                        stream,
                        &mut eredu_runtime::NoopObserver,
                    )
                    .map(|progress| {
                        (
                            match progress.outcome {
                                O::Complete(output) => O::Complete(Some(output)),
                                O::Cancelled => O::Cancelled,
                                O::Unavailable => O::Unavailable,
                            },
                            progress.completed_positions,
                        )
                    })
            } else {
                session
                    .try_prefill_unbudgeted_state_source_cancellable(
                        shape,
                        chunk,
                        make_source,
                        cancellation,
                        stream,
                        &mut eredu_runtime::NoopObserver,
                    )
                    .map(|progress| {
                        (
                            match progress.outcome {
                                O::Complete(()) => O::Complete(None),
                                O::Cancelled => O::Cancelled,
                                O::Unavailable => O::Unavailable,
                            },
                            progress.completed_positions,
                        )
                    })
            };
            result.map_err(|error| {
                Error::after_replicated_model_call(
                    error,
                    before,
                    session.successful_state_restoration_generation(),
                )
            })
        })?;
        let evaluated_tokens =
            usize::try_from(completed).map_err(|error| Error::Other(Box::new(error)))?;
        match outcome {
            O::Cancelled => Ok(P::Cancelled { evaluated_tokens }),
            O::Unavailable => {
                prepared?;
                Err(Error::Speculative(
                    "selected independent composite prefill source is unavailable".into(),
                ))
            }
            O::Complete(output) => Ok(P::Complete(
                eredu_runtime::speculative::autoregressive::AutoregressivePrefill {
                    logits: output.map(|tensor| self.published(tensor.into_array())),
                    evaluated_tokens,
                },
            )),
        }
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

    fn prepare_resident_decoder_copy(
        &self,
    ) -> Result<crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>, Error> {
        super::require_native_control_policy(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        )?;
        self.session
            .inspect_runtime_state(|state| state.prepare_resident_decoder_copy())
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn prepare_original_external_target_source(&self)->Result<(
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ),StartupCause> {
        if A::external_assistant_target_profile_ref(&self.admission).is_none() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound).into());
        }
        if let Some(reason)=super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,D::DISTRIBUTED_PHASE_AGREEMENT,false) {
            return Err(crate::backend::runtime::cache::state::ResidentDecoderPreparationError::Policy(reason).into());
        }
        let origin=self.session.control_state_origin_fixed()?;
        let source=self.session.inspect_runtime_execution_fixed(|_,state,_|
            state.prepare_resident_decoder_copy_fixed())
            .map_err(crate::backend::runtime::cache::state::ResidentDecoderPreparationError::from)??;
        Ok((source,origin))
    }

    fn prepare_original_prediction_target_source(
        &self,
    ) -> Result<(
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ), StartupCause> {
        if !P::present() {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ).into());
        }
        // This is only the target component. The paired startup owner separately
        // constructs prediction state; the full text-snapshot P gate is unchanged.
        if let Some(reason) = super::native_control_policy_rejection(
            D::PARTITIONED_SESSION, D::DISTRIBUTED_PHASE_AGREEMENT, false,
        ) {
            return Err(crate::backend::runtime::cache::state::ResidentDecoderPreparationError::Policy(reason).into());
        }
        let origin = self.session.control_state_origin_fixed()?;
        let source = self.session.inspect_runtime_execution_fixed(|_, state, _| {
            state.prepare_resident_decoder_copy_fixed()
        }).map_err(crate::backend::runtime::cache::state::ResidentDecoderPreparationError::from)??;
        Ok((source, origin))
    }

    fn prepare_resident_decoder_copy_fixed(
        &self,
    ) -> Result<
        crate::backend::runtime::cache::state::PreparedResidentDecoderCopy<'_>,
        crate::backend::runtime::cache::state::ResidentDecoderPreparationError,
    > {
        if let Some(reason) = super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        ) {
            return Err(
                crate::backend::runtime::cache::state::ResidentDecoderPreparationError::Policy(
                    reason,
                ),
            );
        }
        self.session
            .inspect_runtime_execution_fixed(|_, state, _| {
                state.prepare_resident_decoder_copy_fixed()
            })?
    }

    fn resident_copy_input_identity(
        &self,
    ) -> Result<Option<eredu_runtime::SharedPreparedInputCacheIdentity>, Error> {
        // Repeat the same policy, representation and quiescent boundary checks
        // before sharing the actual committed metadata owner.
        self.prepare_resident_decoder_copy()?;
        Ok(self
            .session
            .committed_shared_prompt_input_identity()
            .cloned())
    }

    fn resident_control_origin(
        &self,
    ) -> Result<eredu_runtime::replicated_session::ReplicatedTextControlOrigin, Error> {
        self.prepare_resident_decoder_copy()?;
        self.session
            .control_state_origin()
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn resident_control_origin_fixed(
        &self,
    ) -> Option<
        Result<
            eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
            eredu_runtime::replicated_session::PreparedControlBindingError,
        >,
    > {
        Some(self.session.control_state_origin_fixed())
    }

    fn validate_resident_control_origin(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ) -> Result<(), Error> {
        self.prepare_resident_decoder_copy()?;
        self.session
            .validate_control_state_origin(origin)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    /// Allocation-free origin loan for a saved source. This does not inspect
    /// the installed decoder or confer copy, execution or installation authority.
    fn validate_resident_control_origin_fixed(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
    ) -> Option<Result<(), eredu_runtime::replicated_session::PreparedControlBindingError>> {
        Some(self.session.validate_control_state_origin_fixed(origin))
    }

    fn bind_prepared_dense_control_state(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        state: crate::backend::runtime::cache::state::PublishedDenseResidentKvState,
        prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        super::require_native_control_policy(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        )?;
        let state = MlxHybridState::from_published_dense_control_state(state)?;
        self.session
            .bind_prepared_control_state(origin, state, prompt)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn prepared_resident_control_state_bytes(&self) -> Option<usize> {
        if super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        )
        .is_some()
            || !MlxHybridState::PREPARED_CONTROL_BINDING
        {
            return None;
        }
        crate::composition::mlx::session::MlxNativeTextState::prepared_control_bytes::<MlxHybridState>(
        )
    }

    fn bind_original_resident_control_state(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
        prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<crate::composition::mlx::session::MlxNativeTextState, Error> {
        use crate::composition::mlx::session::{
            MlxNativeTextState, PreparedControlSlotError, prepared_control_slot_error,
        };
        if let Some(reason) = super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        ) {
            return Err(prepared_control_slot_error(
                PreparedControlSlotError::Policy(reason),
            ));
        }
        let state = MlxHybridState::from_published_resident_control_state_fixed(state)
            .map_err(prepared_control_slot_error)?;
        let state = self
            .session
            .bind_prepared_control_state_fixed(origin, state, prompt)
            .map_err(prepared_control_slot_error)?;
        Ok(MlxNativeTextState::from_prepared(state, host))
    }

    fn bind_prepared_resident_control_state(
        &self,
        origin: &eredu_runtime::replicated_session::ReplicatedTextControlOrigin,
        state: crate::backend::runtime::cache::state::PublishedResidentDecoderState,
        prompt: Option<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        super::require_native_control_policy(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        )?;
        let state = MlxHybridState::from_published_resident_control_state(state)?;
        self.session
            .bind_prepared_control_state(origin, state, prompt)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn native_control_support(&self) -> eredu_core::execution_control::ControlSupport {
        use eredu_core::execution_control::ControlSupport;
        let policy = super::native_control_policy_support(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        );
        if matches!(policy, ControlSupport::Unsupported { .. }) {
            return policy;
        }
        if self.session.estimate_control_state().is_none() {
            return ControlSupport::Unsupported {
                reason: "native state isolation or a complete storage estimate is unavailable"
                    .into(),
            };
        }
        ControlSupport::Supported
    }

    fn estimate_original_native_control_state(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        )
        .is_some()
        {
            return None;
        }
        self.session.estimate_original_control_state()
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

    fn exchange_original_control_state(
        &mut self,
        slot: &mut dyn std::any::Any,
        metadata: &eredu_nn::workspace::WorkspaceMetadataFunding,
        media: Option<&eredu_runtime::working_memory::MediaSessionBinding>,
    ) -> Result<Option<eredu_runtime::working_memory::CopiedMediaStateBinding>, Error> {
        use crate::composition::mlx::session::{
            PreparedControlSlotError, prepared_control_slot_error,
        };
        if let Some(reason) = super::native_control_policy_rejection(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        ) {
            return Err(prepared_control_slot_error(
                PreparedControlSlotError::Policy(reason),
            ));
        }
        let slot = slot.downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<MlxHybridState>>()
            .ok_or_else(|| prepared_control_slot_error(PreparedControlSlotError::Type))?;
        self.session
            .exchange_control_state_prepared_fixed(slot, &self.stream, Some(metadata), media)
            .map_err(prepared_control_slot_error)
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

    fn prepare_original_prediction(
        &self,
        context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        P::prepare_original_prediction(self, context)
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
            self.prediction_observers_are_empty = false;
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

    fn supports_external_capture_spans(&self) -> bool {
        // The paired assistant is separately authenticated at construction.
        // A selected partition still needs multi-tensor bundle publication.
        !D::PARTITIONED_SESSION
    }

    fn original_external_prediction_mut(&mut self)
        ->Option<&mut (dyn ErasedExternalPredictionExecutable+'static)> {
        let available=A::external_assistant_target_profile_ref(&self.admission).is_some();
        if available {Some(self as _)}else{None}
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
    fn retained_numeric_state_snapshot(
        &self,
    ) -> Option<Result<RetainedNumericStateSnapshot, Exception>> {
        Some(
            self.session
                .report()
                .map(|report| report.state_report().retained_numeric.clone())
                .map_err(Exception::from_source),
        )
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
        input_identity: eredu_runtime::SharedPreparedInputCacheIdentity,
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
        input_identity: eredu_runtime::SharedPreparedInputCacheIdentity,
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

    fn prefill_cancellable_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Option<Array>, Error> {
        // This loan is distinct from completed prepared media. It comes from
        // the exact resumed quote and confers no B/source-storage credit.
        let continuation = input
            .as_ref()
            .ok()
            .filter(|input| input.original_media().is_none())
            .and_then(|input| input.original_media_metadata());
        if let Some(context) = continuation {
            context
                .charge_metadata(std::mem::size_of::<(
                    Option<eredu_nn::workspace::WorkspaceContext>,
                    Result<Option<Array>, Error>,
                    input::ModelInput<'_>,
                )>())
                .map_err(|cause| Error::Neural(cause.into()))?;
        }
        let continuation_metadata = continuation.cloned();
        let result = (|| {
            let inference_request = input
                .as_ref()
                .ok()
                .and_then(|input| input.inference_request().cloned());
            if let Some(request) = &inference_request {
                request
                    .validate(
                        self.session.inference_execution_identity(),
                        request.geometry(),
                    )
                    .map_err(Error::before_model_mutation)?;
            }
            let max_chunk_positions = input
                .as_ref()
                .ok()
                .and_then(|input| input.prefill_chunk_positions());
            if let Some(packet) = input.as_ref().ok().and_then(|input| input.original_media()) {
                if mask.is_some() {
                    return Err(Error::ArchitectureModel(
                        "explicit composite decoder masks require prepared-input metadata".into(),
                    ));
                }
                if let Some(geometry) = capture_geometry {
                    let [batch, sequence] = packet.shape();
                    geometry
                        .validate_prefill(batch, sequence)
                        .map_err(eredu_nn::Error::backend_source)?;
                }
                let media = self.original_media_prefill.ok_or_else(|| {
                    Error::before_model_mutation(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    )
                })?;
                let identity = input
                    .as_ref()
                    .ok()
                    .and_then(|i| i.shared_cache_identity())
                    .cloned();
                let mut observer = crate::composition::NeutralActivationObserver::new(observer);
                let before = self.session.successful_state_restoration_generation();
                let progress = media(
                    &mut self.session,
                    packet,
                    identity,
                    inference_request.as_ref(),
                    input
                        .as_ref()
                        .ok()
                        .and_then(|input| input.original_media_metadata()),
                    input
                        .as_ref()
                        .ok()
                        .and_then(|input| input.copied_media_semantics()),
                    max_chunk_positions,
                    cancellation,
                    stream,
                    &mut observer,
                )
                .map_err(|error| {
                    Error::after_replicated_model_call(
                        error,
                        before,
                        self.session.successful_state_restoration_generation(),
                    )
                })?;
                return match progress.outcome {
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(output) => {
                        Ok(Some(self.published(output.into_array())))
                    }
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Cancelled => Ok(None),
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Unavailable => {
                        unreachable!("original media cannot silently fall back")
                    }
                };
            }
            let prepared = input.and_then(|input| {
                if mask.is_some() {
                    return Err(match continuation_metadata.as_ref() {
                        Some(context) => Error::Neural(context.metadata_error(format_args!(
                            "explicit composite decoder masks require prepared-input metadata",
                        ))),
                        None => Error::ArchitectureModel(
                            "explicit composite decoder masks require prepared-input metadata"
                                .into(),
                        ),
                    });
                }
                let prepared = match continuation_metadata.as_ref() {
                    Some(context) => decode_input::prepare_continuation::<A>(
                        &self.admission,
                        &self.processor,
                        input,
                        context,
                    )?,
                    None => self.prepare(input)?,
                };
                if let Some(request) = capture_geometry {
                    let [batch, sequence] = prepared.1.decoder_shape();
                    request.validate_prefill(batch, sequence).map_err(|cause| {
                        match continuation_metadata.as_ref() {
                            Some(context) => context.metadata_source(cause),
                            None => eredu_nn::Error::backend_source(cause),
                        }
                    })?;
                }
                Ok(prepared)
            });
            let mut observer = crate::composition::NeutralActivationObserver::new(observer);
            let before = self.session.successful_state_restoration_generation();
            let shape = prepared
                .as_ref()
                .ok()
                .map(|(_, admitted, _)| admitted.decoder_shape());
            #[cfg(test)]
            if prepared.is_ok() {
                crate::tests::support::path_instrumentation::forward();
            }
            // Only the selected typed media visitor installs this entry. Ordinary
            // default ingress stays unchanged; an explicit chunk policy selects the
            // shared retained-source path and cannot fall back after an error.
            let retained_media = prepared.as_ref().is_ok_and(|(input, _, _)| {
                !eredu_architectures::prefill::is_prepared_token_input(input)
            });
            if let Some(media) = self
                .media_prefill
                .filter(|_| max_chunk_positions.is_some() && retained_media)
            {
                let progress = media(
                    &mut self.session,
                    &self.admission,
                    prepared,
                    inference_request.as_ref(),
                    max_chunk_positions,
                    cancellation,
                    stream,
                    &mut observer,
                )
                .map_err(|error| {
                    Error::after_replicated_model_call(
                        error,
                        before,
                        self.session.successful_state_restoration_generation(),
                    )
                })?;
                return match progress.outcome {
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(output) => {
                        Ok(Some(self.published(output.into_array())))
                    }
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Cancelled => Ok(None),
                    eredu_runtime::replicated_session::PrefillSourceOutcome::Unavailable => {
                        unreachable!("typed media entry rejects unavailable source")
                    }
                };
            }
            let mut prepared = Some(prepared);
            match self.session.try_prefill_source_cancellable(
            inference_request.as_ref(), shape, max_chunk_positions,
            |geometry| {
                let Some(Ok((input, _, _))) = prepared.as_ref() else { return Ok(None); };
                if let Some(context) = continuation_metadata.as_ref() {
                    if !eredu_architectures::prefill::is_prepared_token_input(input) {
                        return Ok(None);
                    }
                    let config = A::retain_admission_config_with_metadata(&self.admission, context)?;
                    let Ok((input, _admitted, identity)) = prepared.take().expect("prepared source exists") else {
                        unreachable!("successful preparation checked above")
                    };
                    let source = eredu_architectures::prefill::PreparedCompositeTextPrefill::from_prepared_text_with_metadata(
                        input, geometry, config, input::MlxTensorInputInspector, context,
                    )?;
                    Ok(source.map(|source| match identity {
                        Some(identity) => source.with_shared_cache_identity(identity),
                        None => source,
                    }))
                } else {
                    let Some(Ok((input, _, identity))) = prepared.as_ref() else { unreachable!("checked source") };
                    let source = eredu_architectures::prefill::PreparedCompositeTextPrefill::from_prepared_text(
                        input, geometry, self.admission.clone(), input::MlxTensorInputInspector,
                    )?;
                    Ok(source.map(|source| match identity {
                        Some(identity) => source.with_shared_cache_identity(identity.clone()),
                        None => source,
                    }))
                }
            },
            cancellation, stream, &mut observer,
        ).map_err(|error| Error::after_replicated_model_call(
            error, before, self.session.successful_state_restoration_generation(),
        ))? {
            eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(output) => return Ok(Some(self.published(output.into_array()))),
            eredu_runtime::replicated_session::PrefillSourceOutcome::Cancelled => return Ok(None),
            eredu_runtime::replicated_session::PrefillSourceOutcome::Unavailable => {},
        }
            let prepared = prepared.expect("admitted consumed continuation cannot fall back");
            let (paired, cache_identity) = match prepared {
                Ok((ref prepared, ref admitted, ref identity)) => (
                    match continuation_metadata.as_ref() {
                        Some(context) => {
                            PreparedCompositeInput::new_with_metadata(prepared, admitted, context)
                        }
                        None => PreparedCompositeInput::new(prepared, admitted)
                            .map_err(eredu_nn::Error::backend),
                    },
                    identity.clone(),
                ),
                Err(error) => (
                    Err(match continuation_metadata.as_ref() {
                        Some(context) => context.metadata_source(error),
                        None => eredu_nn::Error::backend_source(error),
                    }),
                    None,
                ),
            };
            let output = self
                .session
                .prefill_input_result_with_shared_identity(
                    paired,
                    cache_identity,
                    stream,
                    &mut observer,
                )
                .map(MlxTensor::into_array)
                .map_err(|error| {
                    Error::after_replicated_model_call(
                        error,
                        before,
                        self.session.successful_state_restoration_generation(),
                    )
                })?;
            Ok(Some(self.published(output)))
        })();
        // All local source/chunk and native completion owners retire before the
        // escaped cause takes its independent account-only funding alias.
        result.map_err(|cause| {
            match continuation_metadata
                .as_ref()
                .and_then(|context| context.metadata_funding())
            {
                Some(funding) => {
                    crate::composition::mlx::model::retain_planning_error(cause, funding)
                }
                None => cause,
            }
        })
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        self.decode_result_with_observer_and_metadata(tokens, stream, observer, None)
    }

    fn retain_continuation_metadata(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Option<eredu_nn::workspace::WorkspaceContext> {
        Some(context.clone())
    }

    fn decode_result_with_observer_and_metadata(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Array, Error> {
        let prepared = tokens.and_then(|tokens| {
            decode_input::prepare::<A>(&self.admission, &self.processor, tokens, metadata)
        });
        let paired = match prepared {
            Ok((ref prepared, ref admitted, _)) => match metadata {
                Some(metadata) => {
                    PreparedCompositeInput::new_with_metadata(prepared, admitted, metadata)
                }
                None => PreparedCompositeInput::new(prepared, admitted)
                    .map_err(eredu_nn::Error::backend),
            },
            Err(error) => Err(match metadata {
                Some(metadata) => metadata.metadata_source(error),
                None => eredu_nn::Error::backend_source(error),
            }),
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

    fn prefill_external_prediction_spans(
        &mut self, input: crate::composition::mlx::MlxModelInput,
        request: &ExternalPredictionCaptureRequest, cache: &mut MlxPredictionTargetState,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<MlxTensor,Error>,
        cancellation: &eredu_core::GenerationCancellationToken,
        context: crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_runtime::replicated_session::PrefillSourceProgress<Option<MlxTensor>>,Error> {
        external_prefill::run::<A,D>(&mut self.session,&self.admission,&self.processor,
            input,request,cache,receiver,cancellation,context)
    }

    fn verify_external_prediction_target_with_evidence(
        &mut self, tokens: &MlxTensor, request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
        context: crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    ) -> Result<eredu_architectures::external_assistant::ExternalTargetResult<MlxTensor>, Error> {
        use super::super::prediction::phase::external_target::{self, Output};
        use eredu_runtime::speculative::external_occurrence::ExternalInvocationKind;
        let admission=&self.admission;
        let processor=&self.processor;
        let stream=&self.stream;
        let output=external_state::with_state::<A,D,_>(&mut self.session,cache,stream,Some(context),
            |session,prior| external_target::run::<A,D,_>(session,tokens,Some(request),
                ExternalInvocationKind::TargetVerification,None,eredu_core::OutputDemand::Sequence,
                context,prior,|session,capture,metadata|{
                    let capture=capture.ok_or(Error::PrefillScopeUnavailable)?;
                    let (prepared,admitted,_)=decode_input::prepare::<A>(admission,processor,tokens.as_array(),metadata)?;
                    let paired=match metadata {
                        Some(metadata)=>PreparedCompositeInput::new_with_metadata(&prepared,&admitted,metadata)
                            .map_err(Error::Neural)?,
                        None=>PreparedCompositeInput::new(&prepared,&admitted).map_err(Error::ArchitectureModel)?,
                    };
                    let mut observer=capture.observer();
                    let (scores,captured)=session.decode_input_with_capture(paired,stream,&mut observer,|forward|{
                        let values=capture.values()?;
                        let result=match metadata {
                            Some(metadata)=>A::external_prediction_capture_with_metadata(request,forward,values,metadata),
                            None=>A::external_prediction_capture(request,forward,values),
                        }?;
                        result.ok_or_else(||match metadata {
                            Some(metadata)=>metadata.metadata_error(format_args!("target omitted its selected assistant capture")),
                            None=>eredu_nn::Error::backend("target omitted its selected assistant capture"),
                        })
                    }).map_err(|cause|external_state::failure(cause,Some(context)))?;
                    Ok(Output{scores:Some(scores),capture:Some(captured),frontier:0,commit:None,evidence:None})
                })
        )?;
        Ok(eredu_architectures::external_assistant::ExternalTargetResult {
            logits:output.scores.ok_or(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch))?,
            capture:output.capture.ok_or(Error::PrefillScopeUnavailable)?,evidence:output.evidence,
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

    fn apply_external_prediction_target_operation_with_source(
        &mut self,operation:ExternalPredictionTargetOperation<'_,MlxTensor>,
        context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    )->Result<eredu_architectures::speculative_execution::EmbeddedPredictionTensor<MlxTensor>,Error>{
        use super::super::prediction::phase::external_target::{self,Output};
        use eredu_runtime::speculative::external_occurrence::ExternalInvocationKind as Kind;
        let (input,kind)=match operation {
            ExternalPredictionTargetOperation::TokenEmbeddings(input)=>(input,Kind::TargetTokenEmbeddings),
            ExternalPredictionTargetOperation::ProjectLogits(input)=>(input,Kind::TargetProjectLogits),
            _=>return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)),
        };
        let mut prior=None;
        let output=external_target::run::<A,D,_>(&mut self.session,input,None,kind,None,
            eredu_core::OutputDemand::Sequence,context,&mut prior,|session,_,_|{
                let value=session.apply_prediction_target_operation(
                    CompositePredictionTargetOperation{operation},context.target())
                    .map_err(|cause|external_state::failure(cause,Some(context)))?;
                Ok(Output{scores:Some(value),capture:None,frontier:0,commit:None,evidence:None})
            })?;
        output.into_tensor(context)
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
    fn retained_storage(
        &self,
    ) -> Result<crate::backend::runtime::residency::storage::RetainedStorage, Error> {
        let mut storage = crate::backend::runtime::residency::storage::RetainedStorage::default();
        <Self as ReplicatedPredictionCapability<A, S, D>>::collect_retained_storage(
            self,
            &mut storage,
        )?;
        Ok(storage)
    }

    /// Fills caller-owned storage without allocating an intermediate inventory.
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        super::super::prediction::parameters::collect_retained_storage::<A, P>(
            &self.extension,
            storage,
        )
    }

    fn count_parameter_owners(
        &self,
        counts: &mut ParameterOwnerCounts,
        guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        super::super::prediction::parameters::count_parameter_owners::<A, P>(
            &self.extension,
            counts,
            guard,
        )
    }

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

    fn prepare_original_prediction(
        model: &CompletedReplicatedText<A, S, D, Self>,
        context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        Some(context.prepare::<A, P>(
            &model.prediction.extension,
            &model.prediction.selected,
            model.session.control_state_origin_fixed(),
            &model.stream,
        ))
    }

    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let controls = [
            std::mem::size_of::<eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxReplicatedTextMechanisms<A, S>,
                D,
                P,
                super::prepared_speculative::MlxTextPredictionInput,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
                crate::composition::mlx::replicated_text::prediction::phase::MlxPredictionPhase, >>(),
            std::mem::size_of::<eredu_architectures::speculative_execution::EmbeddedPredictionExecutor<
                eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxReplicatedTextMechanisms<A, S>,
                D,
                P,
                super::prepared_speculative::MlxTextPredictionInput,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
                crate::composition::mlx::replicated_text::prediction::phase::MlxPredictionPhase, >, super::prepared_speculative::MlxEmbeddedPredictionMechanisms>>(),
            std::mem::size_of::<eredu_architectures::speculative_execution::DynEmbeddedExecutor<
                super::prepared_speculative::MlxEmbeddedExecutorTypes>>(),
            std::mem::size_of::<MlxEmbeddedPredictionObservers>(),
            std::mem::size_of::<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>>(),
        ];
        if let Err(cause) = continuation.construction_controls(
            controls
                .into_iter()
                .try_fold(std::mem::size_of_val(&controls), usize::checked_add),
        ) {
            return Some(Err(cause));
        }
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
                crate::composition::mlx::replicated_text::prediction::phase::MlxPredictionPhase,
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
