use super::*;
use eredu_nn::Tensor as _;
use ref_cast::RefCast;

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

impl<A, S> CompletedReplicatedText<A, S>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    pub(super) fn new(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        stream: &Stream,
        weights_stream: &Stream,
    ) -> Result<Self, Error> {
        Self::new_with_residency(prepared, store, stream, weights_stream, Default::default())
    }

    pub(super) fn new_with_residency(
        prepared: PreparedReplicatedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
        stream: &Stream,
        weights_stream: &Stream,
        residency: super::super::prediction::parameters::PredictionResidency,
    ) -> Result<Self, Error> {
        Self::new_with_prepared_layerwise(prepared, store, stream, weights_stream, residency, None)
    }

    pub(super) fn new_with_prepared_layerwise(
        prepared: PreparedReplicatedTextArchitecture<A>,
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
            prediction_observers_are_empty: true,
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
            .map_err(eredu_nn::Error::backend_retained_source)?;
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
            prediction_observers_are_empty: self.prediction_observers_are_empty,
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
    S: MlxStateMechanisms + 'static,
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
    fn autoregressive_forward_inner(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        prefill: bool,
        demand: eredu_core::OutputDemand,
        stream: &Stream,
        completion: Option<&mut dyn AutoregressiveSequenceCompletion>,
    ) -> Result<Option<Array>, Error> {
        let funding = completion
            .as_ref()
            .map(|completion| completion.metadata_funding());
        fn retain<E: std::error::Error + Send + Sync + 'static>(
            cause: E,
            funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
        ) -> Error {
            match funding {
                Some(funding) => Error::Neural(funding.metadata_source(cause)),
                None => Error::Other(Box::new(cause)),
            }
        }
        if !cache.is::<S>() {
            if funding.is_some() {
                return Err(Error::PrefillControl(
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                ));
            }
            return Err(Error::Speculative(
                "ordinary lane state type differs".into(),
            ));
        }
        let mut completion = completion;
        let checkpoint = match completion.as_mut() {
            Some(completion) => {
                let mut checkpoint = completion.take_checkpoint()?;
                if !checkpoint.is::<S>() {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
                Some(super::mechanisms::StateCheckpoint::independent(
                    checkpoint.take_state::<S>()?,
                ))
            }
            None => None,
        };
        let lane = cache
            .state_mut::<S>()
            .expect("prediction lane type checked");
        if let Err(error) = self.session.exchange_prediction_target_state(lane, stream) {
            return Err(retain(error, funding.as_ref()));
        }
        let tokens = MlxTensor::ref_cast(tokens);
        let pass = if prefill {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let result = match completion {
            Some(completion) if prefill => {
                self.session.prefill_span_with_checkpoint_and_completion(
                    A::text_input(tokens, None),
                    demand,
                    stream,
                    checkpoint.expect("original completion owns prepared checkpoint"),
                    funding
                        .as_ref()
                        .expect("original completion retains funding"),
                    |output, state, stream| {
                        completion.complete(output.map(MlxTensor::as_array), state, stream)
                    },
                )
            }
            Some(completion) => self
                .session
                .sequence_logits_with_checkpoint_and_completion(
                    A::text_input(tokens, None),
                    pass,
                    stream,
                    checkpoint.expect("original completion owns prepared checkpoint"),
                    |output, state, stream| {
                        completion.complete(Some(output.as_array()), state, stream)
                    },
                )
                .map(Some),
            None => self
                .session
                .sequence_logits(A::text_input(tokens, None), pass, stream)
                .map(Some),
        };
        let restored = match self.session.exchange_prediction_target_state(lane, stream) {
            Ok(()) => Ok(()),
            Err(error) => super::finish_prediction_state_operation_with_metadata(
                Err(retain(error, funding.as_ref())),
                self.session
                    .recover_prediction_target_state_after_failure(lane)
                    .map_err(|recovery| retain(recovery, funding.as_ref())),
                funding.as_ref(),
            ),
        };
        let output = super::finish_prediction_state_operation_with_metadata(
            result.map_err(|e| retain(e, funding.as_ref())),
            restored,
            funding.as_ref(),
        )?;
        Ok(self.published(output.map(MlxTensor::into_array)))
    }

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
    fn inference_execution_identity(
        &self,
    ) -> &eredu_runtime::working_memory::InferenceExecutionIdentity {
        self.session.inference_execution_identity()
    }

    fn uses_ordinary_unit_equations(&self) -> bool {
        self.session.execution_strategy().uses_ordinary_unit_equations()
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
            .inspect_runtime_state(|state| state.collect_retained_storage(storage))
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
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::inspection_adapters_for_test(&self.session, geometry, pool)
    }
    #[cfg(test)]
    fn prefill_status_for_test(&self) -> Result<(bool, bool), Error> {
        MlxReplicatedTextMechanisms::prefill_status_for_test(&self.session)
    }

    fn bind_layerwise_neural_recipe(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::bind_session_neural_recipe(&self.session, pool, recipe, funding)
    }

    fn bind_speculative_neural_recipe(
        &self,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        recipe: &mut crate::backend::nn::workspace::AutoregressiveEquationRecipe,
    ) -> Result<
        (
            u64,
            Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
        ),
        Error,
    > {
        MlxReplicatedTextMechanisms::bind_session_speculative_neural_recipe(
            &self.session,
            source,
            pool,
            funding,
            recipe,
        )
    }
    fn prepare_speculative_neural_bank(
        &self,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        role: eredu_runtime::working_memory::OriginalSpeculativeRole,
        scope: &safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    {
        MlxReplicatedTextMechanisms::prepare_session_speculative_neural_bank(
            &self.session,
            source,
            recipe,
            role,
            scope,
            partition,
        )
    }

    fn prepare_speculative_span_neural_bank(
        &self,
        source: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        recipe: &crate::backend::nn::workspace::AutoregressiveEquationRecipe,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        scope: &safemlx::SubmissionScope,
        partition:Option<crate::backend::runtime::execution::generic::SpeculativeSourcePartition<'_>>,
    ) -> Result<Option<crate::backend::runtime::execution::generic::SpeculativeNeuralOwner>, Error>
    {
        MlxReplicatedTextMechanisms::prepare_session_speculative_span_neural_bank(
            &self.session,
            source,
            recipe,
            span,
            scope,
            partition,
        )
    }

    fn prefill_control_facts(
        &self,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
        geometry: eredu_core::InferenceGeometry,
        graph_capacity: std::num::NonZeroU64,
        native_root_capacity: Option<u64>,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        native_recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
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
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
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
        MlxReplicatedTextMechanisms::retire_session_controls(&self.session)
    }

    #[cfg(test)]
    fn opening_rows_status_for_test(&self) -> Result<(bool, bool), Error> {
        MlxReplicatedTextMechanisms::opening_rows_status_for_test(&self.session)
    }

    #[cfg(test)]
    fn busy_rows_retirement_for_test(&self) -> Result<(), Error> {
        MlxReplicatedTextMechanisms::busy_rows_retirement_for_test(&self.session)
    }

    fn shared_observation_paths(&self) -> Option<&eredu_runtime::SharedLayeredObservationPaths> {
        self.session.shared_observation_paths()
    }

    fn validate_prepared_observation_paths(
        &self,
        expected: &eredu_runtime::SharedLayeredObservationPaths,
        metadata: eredu_runtime::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<(), Error> {
        self.session
            .validate_prepared_observation_paths(expected)
            .map_err(|cause| match metadata.funding() {
                Some(funding) => crate::composition::mlx::model::retain_planning_error(cause, funding),
                None => Error::Other(Box::new(cause)),
            })
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

    fn prepared_layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error> {
        let frames = [
            std::mem::size_of::<eredu_runtime::replicated_session::RuntimeInspectionBoundary>(),
            std::mem::size_of::<
                Result<
                    Result<crate::backend::runtime::execution::generic::LayerwiseWorkspace, Error>,
                    eredu_runtime::replicated_session::RuntimeInspectionBoundary,
                >,
            >(),
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                    ))?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        self.session
            .inspect_runtime_execution_fixed(|_, _, runtime| {
                D::resident_policy(runtime)
                    .or_else(|| D::bounded_policy(runtime))
                    .ok_or(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    ))?
                    .prepared_layerwise_workspace(allocation, context)
            })
            .map_err(Error::RuntimeInspection)?
    }

    fn layerwise_workspace(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
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
                state.project_resident_workspace_with_storage(batch, context)
            })
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn original_text_frontier(&self) -> Result<Option<u64>, super::super::prediction::OriginalTextFrontierError> {
        use eredu_runtime::ReplicatedTextSessionMechanisms;
        use super::super::prediction::OriginalTextFrontierError;
        self.session.inspect_runtime_execution_fixed(|mechanisms, state, _| {
            mechanisms.original_prefill_state_frontier(state)
        }).map_err(OriginalTextFrontierError::Boundary)?
          .map_err(OriginalTextFrontierError::Mechanism)
    }

    fn validate_text_frontier(&self, expected: u64) -> Result<(), Error> {
        self.session
            .inspect_runtime_state(|state| state.validate_text_frontier(expected))
            .map_err(|error| Error::Other(Box::new(error)))
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
        if !cache.is::<S>() {
            return Err(Error::Speculative(
                "ordinary lane state type differs".into(),
            ));
        }
        let chunk = input.prefill_chunk_positions();
        let tokens = input::text_token_ids(input, stream)
            .map(MlxTensor::from_array)
            .map_err(Error::from);
        let shape = tokens.as_ref().ok().and_then(|tokens| {
            let [batch, count] = tokens.shape() else {
                return None;
            };
            Some([u64::try_from(*batch).ok()?, u64::try_from(*count).ok()?])
        });
        let mut lane = cache.take_state::<S>()?;
        if let Err(error) = self
            .session
            .exchange_prediction_target_state(&mut lane, stream)
        {
            cache.restore_state(lane);
            return Err(Error::Other(Box::new(error)));
        }
        let before = self.session.successful_state_restoration_generation();
        let make_source = |geometry| {
            let Ok(tokens) = &tokens else {
                return Ok(None);
            };
            // Lane state has separate ownership. Do not install its prompt
            // identity onto the temporarily borrowed canonical model session.
            eredu_architectures::prefill::PreparedTextPrefill::from_tensor(
                tokens.clone(),
                None,
                geometry,
            )
            .map(Some)
        };
        let result = if sample {
            self.session
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
            self.session
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
        }
        .map_err(|error| {
            Error::after_replicated_model_call(
                error,
                before,
                self.session.successful_state_restoration_generation(),
            )
        });
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
        let (outcome, completed) = super::finish_prediction_state_operation(result, restored)?;
        let evaluated_tokens =
            usize::try_from(completed).map_err(|error| Error::Other(Box::new(error)))?;
        match outcome {
            O::Cancelled => Ok(P::Cancelled { evaluated_tokens }),
            O::Unavailable => {
                tokens?;
                Err(Error::Speculative(
                    "selected independent text prefill source is unavailable".into(),
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
        self.autoregressive_forward_inner(
            tokens,
            cache,
            prefill,
            eredu_core::OutputDemand::Sequence,
            stream,
            None,
        )
        .map(|output| output.expect("sequence forward preserves output"))
    }
    fn autoregressive_forward_with_completion(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        stream: &Stream,
        completion: &mut dyn AutoregressiveSequenceCompletion,
    ) -> Result<Array, Error> {
        self.autoregressive_forward_inner(
            tokens,
            cache,
            false,
            eredu_core::OutputDemand::Sequence,
            stream,
            Some(completion),
        )
        .map(|output| output.expect("sequence forward preserves output"))
    }

    fn autoregressive_prefill_span_with_completion(
        &mut self,
        tokens: &Array,
        cache: &mut MlxPredictionTargetState,
        span: &eredu_runtime::working_memory::OriginalSpeculativePrefillSpan,
        stream: &Stream,
        completion: &mut dyn AutoregressiveSequenceCompletion,
    ) -> Result<Option<Array>, Error> {
        if !span.role().same_role(completion.role())
            || cache.generation_fixed() != Some(span.chunk().position)
        {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        self.autoregressive_forward_inner(
            tokens,
            cache,
            true,
            span.chunk().output,
            stream,
            Some(completion),
        )
    }

    fn resident_reset_profile(&self) -> Option<ResidentResetProfile> {
        if P::present() {
            None
        } else {
            S::resident_reset_profile()
        }
    }

    fn resident_reset_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetSource<'_, MlxKeyValueState>,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        if P::present() {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound);
        }
        self.session
            .projected_resident_reset_source::<MlxKeyValueState>()
    }

    fn install_resident_reset(
        &mut self,
        installation: eredu_runtime::working_memory::ResidentResetInstallation<MlxKeyValueState>,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetDisplaced<MlxKeyValueState>,
        (
            eredu_runtime::working_memory::WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<MlxKeyValueState>,
        ),
    > {
        if P::present() {
            return Err((
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                installation,
            ));
        }
        self.session.install_resident_reset(installation)
    }

    fn resident_hybrid_reset_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetSource<'_, MlxHybridState>,
        eredu_runtime::working_memory::WorkingMemoryError,
    > {
        if P::present() {
            return Err(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound);
        }
        self.session
            .projected_resident_reset_source::<MlxHybridState>()
    }

    fn install_resident_hybrid_reset(
        &mut self,
        installation: eredu_runtime::working_memory::ResidentResetInstallation<MlxHybridState>,
    ) -> Result<
        eredu_runtime::working_memory::ResidentResetDisplaced<MlxHybridState>,
        (
            eredu_runtime::working_memory::WorkingMemoryError,
            eredu_runtime::working_memory::ResidentResetInstallation<MlxHybridState>,
        ),
    > {
        if P::present() {
            return Err((
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                installation,
            ));
        }
        self.session.install_resident_reset(installation)
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
        let state = S::from_published_dense_control_state(state)?;
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
            || !S::PREPARED_CONTROL_BINDING
        {
            return None;
        }
        crate::composition::mlx::session::MlxNativeTextState::prepared_control_bytes::<S>()
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
        let state = S::from_published_resident_control_state_fixed(state)
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
        let state = S::from_published_resident_control_state(state)?;
        self.session
            .bind_prepared_control_state(origin, state, prompt)
            .map(|state| Box::new(state) as Box<dyn std::any::Any>)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn native_control_support(&self) -> eredu_core::execution_control::ControlSupport<&'static str> {
        use eredu_core::execution_control::ControlSupport;
        let policy = super::native_control_policy_support(
            D::PARTITIONED_SESSION,
            D::DISTRIBUTED_PHASE_AGREEMENT,
            P::present(),
        );
        if matches!(policy, ControlSupport::Unsupported { .. }) {
            return policy;
        }
        if self.session.estimate_original_control_state().is_none() {
            return ControlSupport::Unsupported {
                reason: "native state isolation or a complete storage estimate is unavailable",
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
            return Err(Error::ArchitectureModel(reason.into()));
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
            return Err(Error::ArchitectureModel(reason.into()));
        }
        let saved = saved
            .downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| Error::ArchitectureModel("native control state type differs".into()))?;
        self.session
            .validate_control_state(saved)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn original_control_branch_sources(&self, slot: &dyn std::any::Any)
        -> Result<[eredu_runtime::replicated_session::ControlBranchSource; 2], Error> {
        use crate::composition::mlx::session::{PreparedControlSlotError, prepared_control_slot_error};
        let slot = slot.downcast_ref::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| prepared_control_slot_error(PreparedControlSlotError::Type))?;
        self.session.original_control_branch_sources(slot).map_err(prepared_control_slot_error)
    }

    fn exchange_original_control_state(
        &mut self,
        slot: &mut dyn std::any::Any,
        metadata: &eredu_nn::workspace::HostMetadataFunding,
        media: Option<&eredu_runtime::working_memory::MediaSessionBinding>,
        branch: Option<(&eredu_runtime::working_memory::PendingTextBranchExchange,
            &[eredu_runtime::replicated_session::ControlBranchSource; 2])>,
    ) -> Result<eredu_runtime::replicated_session::ControlExchangeResult, Error> {
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
        let slot = slot
            .downcast_mut::<eredu_runtime::replicated_session::ReplicatedTextControlState<S>>()
            .ok_or_else(|| prepared_control_slot_error(PreparedControlSlotError::Type))?;
        self.session
            .exchange_control_state_prepared_fixed(slot, &self.stream, Some(metadata), media, branch)
            .map_err(prepared_control_slot_error)
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
            .forward(MlxTensor::ref_cast(tokens), None, stream)
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
        let tokens = MlxTensor::ref_cast(tokens);
        let mask = mask.map(MlxTensor::ref_cast);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let output = self
            .session
            .forward_with_observer(tokens, mask, stream, &mut observer)
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

    fn prefill_cancellable_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        cancellation: &eredu_core::GenerationCancellationToken,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Option<Array>, Error> {
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
        let cache_identity = input.as_ref().ok().and_then(|input| {
            input.shared_cache_identity().cloned().or_else(|| {
                input
                    .cache_identity()
                    .cloned()
                    .map(eredu_runtime::SharedPreparedInputCacheIdentity::new)
            })
        });
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
                        .map_err(eredu_nn::Error::backend_retained_source)?;
                }
                Ok(tokens)
            })
            .map(MlxTensor::from_array);
        let mask = mask.cloned().map(MlxTensor::from_array);
        let mut observer = crate::composition::NeutralActivationObserver::new(observer);
        let before = self.session.successful_state_restoration_generation();
        let shape = tokens.as_ref().ok().and_then(|tokens| {
            let [batch, sequence] = tokens.shape() else {
                return None;
            };
            Some([u64::try_from(*batch).ok()?, u64::try_from(*sequence).ok()?])
        });
        #[cfg(test)]
        if tokens.is_ok() {
            crate::tests::support::path_instrumentation::forward();
        }
        match self
            .session
            .try_prefill_source_cancellable(
                inference_request.as_ref(),
                shape,
                max_chunk_positions,
                |geometry| {
                    let Ok(tokens) = &tokens else {
                        return Ok(None);
                    };
                    let mut source =
                        eredu_architectures::prefill::PreparedTextPrefill::from_tensor(
                            tokens.clone(),
                            mask.clone(),
                            geometry,
                        )?;
                    if let Some(identity) = &cache_identity {
                        source = source.with_shared_cache_identity(identity.clone());
                    }
                    Ok(Some(source))
                },
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
            })? {
            eredu_runtime::replicated_session::PrefillSourceOutcome::Complete(output) => {
                return Ok(Some(self.published(output.into_array())));
            }
            eredu_runtime::replicated_session::PrefillSourceOutcome::Cancelled => return Ok(None),
            eredu_runtime::replicated_session::PrefillSourceOutcome::Unavailable => {}
        }
        let input = match tokens {
            Ok(ref tokens) => Ok(A::text_input(tokens, mask.as_ref())),
            Err(error) => Err(eredu_nn::Error::backend_retained_source(error)),
        };
        let output = self
            .session
            .prefill_input_result_with_shared_identity(input, cache_identity, stream, &mut observer)
            .map(MlxTensor::into_array)
            .map_err(|error| {
                Error::after_replicated_model_call(
                    error,
                    before,
                    self.session.successful_state_restoration_generation(),
                )
            })?;
        Ok(Some(self.published(output)))
    }

    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        let input = match tokens {
            Ok(tokens) => Ok(A::text_input(MlxTensor::ref_cast(tokens), None)),
            Err(error) => Err(eredu_nn::Error::backend_retained_source(error)),
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

use eredu_nn::workspace::WorkspaceMetadataAllocation;
