use super::*;

fn finish_routed_composite_session<A, D, F>(
    (stream, finalizer): (&Stream, F),
    session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    facts: eredu_architectures::prepared_execution::PreparedCompositeSessionFacts<
        A::AdmissionConfig,
    >,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
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
    F: super::partitioned::CompositeExecutableFinalizer<A>,
{
    finalizer.finish(complete_routed_composite_session(stream, session, facts)?)
}

fn complete_routed_composite_session<A, D>(
    stream: &Stream,
    session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    facts: eredu_architectures::prepared_execution::PreparedCompositeSessionFacts<
        A::AdmissionConfig,
    >,
) -> Result<CompletedComposite<A, D>, Error>
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
{
    let banks = session.execution_strategy().parameter_banks()?;
    let (text, processor, admission) = facts.into_parts();
    let (identity, capability, model_type, residency) = text.into_parts();
    Ok(
        CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
            session, admission, processor, identity, capability, model_type, residency, None, None,
            None, true, stream,
        )
        .with_parameter_banks(banks)?,
    )
}

fn finish_routed_media_session<A, D>(
    stream: &Stream,
    session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
        D,
    >,
    facts: eredu_architectures::prepared_execution::PreparedCompositeSessionFacts<
        A::AdmissionConfig,
    >,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: eredu_architectures::composite_execution::CompositeMediaIngressArchitecture<
            MlxNeuralBackend,
            MlxHybridState,
            Error = eredu_nn::Error,
        > + 'static,
    A::InputPartPlan: 'static,
    A::AdmissionConfig: 'static,
    A::Error: std::fmt::Display,
    D: eredu_runtime::media_prefill::MediaTextExecutionStrategy<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxHybridState,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
            MlxArchitectureLayerwisePolicy<PreparedCompositeArchitecture<A>, MlxHybridState>,
        > + MlxParameterBankTelemetry
        + 'static,
{
    Ok(Box::new(
        complete_routed_composite_session(stream, session, facts)?.with_media_prefill(),
    ))
}

/// Family-agnostic MLX binder for architecture-owned composite ingress.
#[derive(Clone, Copy)]
pub(crate) struct CompositeBindingVisitor<'a> {
    pub(in crate::composition::mlx) addressable_manager: Option<&'a AddressableManagerSlot>,
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
    pub construction_sources: Option<&'a NativeConstructionSlot>,
}

impl<A, D, P> CompositePredictionCapability<A, D> for SelectedPrediction<P>
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
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            PreparedCompositeArchitecture<A>,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    fn collect_retained_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        super::super::prediction::parameters::collect_retained_storage::<
            PreparedCompositeArchitecture<A>,
            P,
        >(&self.extension, storage)
    }

    fn count_parameter_owners(
        &self,
        counts: &mut ParameterOwnerCounts,
        guard: &mut safemlx::RuntimeCallGuard,
    ) -> Result<(), ParameterOwnerSourceError> {
        super::super::prediction::parameters::count_parameter_owners::<
            PreparedCompositeArchitecture<A>,
            P,
        >(&self.extension, counts, guard)
    }

    fn visit_parameter_publication(
        &mut self,
        visitor: &mut dyn eredu_runtime::parameter_operations::ParameterPublication<MlxTensor>,
    ) -> Result<bool, Error> {
        super::super::prediction::parameters::visit_publication::<
            PreparedCompositeArchitecture<A>,
            P,
        >(&mut self.extension, visitor);
        Ok(true)
    }
    fn visit_parameter_slots(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) {
        super::super::prediction::parameters::visit::<PreparedCompositeArchitecture<A>, P>(
            &mut self.extension,
            visitor,
        );
    }
    fn with_parameter_slots(
        &mut self,
        module: usize,
        operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,

        preparation: Option<
            &crate::backend::runtime::execution::generic::MlxParameterPreparation<'_>,
        >,
    ) -> Result<bool, Error> {
        super::super::prediction::parameters::with_slots::<PreparedCompositeArchitecture<A>, P>(
            &mut self.extension,
            module,
            operation,
            preparation,
        )
    }
    fn activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        self.extension.activation_execution(&self.selected)
    }

    fn prepare_original_prediction(
        model: &CompletedComposite<A, D, Self>,
        context: &mut OriginalPredictionStartupContext<'_>,
    ) -> Option<Result<OriginalPredictionLane, StartupCause>> {
        Some(context.prepare::<PreparedCompositeArchitecture<A>, P>(
            &model.prediction.extension,
            &model.prediction.selected,
            model.session.control_state_origin_fixed(),
            &model.stream,
        ))
    }

    fn lend(
        model: &mut CompletedComposite<A, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let input = MlxCompositePredictionInput::<A> {
            admission: &model.admission,
            processor: &model.processor,
        };
        let controls = [
            std::mem::size_of::<eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
                P,
                MlxCompositePredictionInput<'_, A>,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
                crate::composition::mlx::replicated_text::prediction::phase::MlxPredictionPhase, >>(),
            std::mem::size_of::<eredu_architectures::speculative_execution::EmbeddedPredictionExecutor<
                eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy<
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
                P,
                MlxCompositePredictionInput<'_, A>,
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
                PreparedCompositeArchitecture<A>,
                MlxNeuralBackend,
                MlxHybridState,
                MlxReplicatedTextMechanisms<PreparedCompositeArchitecture<A>, MlxHybridState>,
                D,
                P,
                MlxCompositePredictionInput<'_, A>,
                MlxEmbeddedPredictionMaterializer,
                super::prepared_speculative::MlxEmbeddedPredictionMechanisms,
                crate::composition::mlx::replicated_text::prediction::phase::MlxPredictionPhase,
            >::new(
                &mut model.session,
                &mut model.prediction.extension,
                selected,
                input,
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

impl
    eredu_architectures::replicated_text::CompositePredictionTargetVisitor<
        MlxNeuralBackend,
        MlxHybridState,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<MlxNeuralBackend>>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        let mut prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        let residency = super::super::prediction::parameters::residency::<
            PreparedCompositeArchitecture<A>,
            _,
        >(&mut prediction.extension)?;
        CompletedComposite::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            residency,
            self.construction_sources.and_then(std::cell::Cell::take),
        )?
        .with_prediction(prediction, self.capability)
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<MlxNeuralBackend>>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
        PreparedCompositeArchitecture<A>:
            eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            >,
    {
        let mut mechanisms = MlxReplicatedTextMechanisms::<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        >::new(store.clone(), self.stream, self.weights_stream)?;
        let mut prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        mechanisms.set_prediction_residency(super::super::prediction::parameters::residency::<
            PreparedCompositeArchitecture<A>,
            _,
        >(&mut prediction.extension)?);
        mechanisms
            .set_prepared_construction_sources(self.construction_sources.and_then(std::cell::Cell::take));
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.stream,
            |banks, options| {
                super::routed::selected_addressable_banks(
                    banks,
                    store.clone(),
                    options,
                    self.weights_stream,
                    self.stream,
                    self.addressable_manager,
                )
            },
            (
                self.stream,
                PredictionReplicatedFinalizer {
                    prediction,
                    capability: self.capability,
                },
            ),
            finish_routed_composite_session,
            finish_routed_composite_session,
        )
        .map_err(super::routed::construction_error)
    }
}

impl CompositeTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for CompositeBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
    {
        CompletedComposite::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            Default::default(),
            self.construction_sources.and_then(std::cell::Cell::take),
        )
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mut mechanisms: MlxReplicatedTextMechanisms<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        > = MlxReplicatedTextMechanisms::new(store.clone(), self.stream, self.weights_stream)?;
        mechanisms
            .set_prepared_construction_sources(self.construction_sources.and_then(std::cell::Cell::take));
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.stream,
            |banks, options| {
                super::routed::selected_addressable_banks(
                    banks,
                    store.clone(),
                    options,
                    self.weights_stream,
                    self.stream,
                    self.addressable_manager,
                )
            },
            (self.stream, OrdinaryReplicatedFinalizer),
            finish_routed_composite_session,
            finish_routed_composite_session,
        )
        .map_err(super::routed::construction_error)
    }
    fn visit_media<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_architectures::composite_execution::CompositeMediaIngressArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
    {
        CompletedComposite::new_with_construction_sources(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            Default::default(),
            self.construction_sources.and_then(std::cell::Cell::take),
        )
        .map(|model| {
            Box::new(model.with_media_prefill()) as Box<dyn ErasedReplicatedTextExecutable>
        })
    }

    fn visit_routed_media<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_architectures::composite_execution::CompositeMediaIngressArchitecture<
                MlxNeuralBackend,
                MlxHybridState,
            > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mut mechanisms: MlxReplicatedTextMechanisms<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        > = MlxReplicatedTextMechanisms::new(store.clone(), self.stream, self.weights_stream)?;
        mechanisms
            .set_prepared_construction_sources(self.construction_sources.and_then(std::cell::Cell::take));
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.stream,
            |banks, options| {
                super::routed::selected_addressable_banks(
                    banks,
                    store.clone(),
                    options,
                    self.weights_stream,
                    self.stream,
                    self.addressable_manager,
                )
            },
            self.stream,
            finish_routed_media_session,
            finish_routed_media_session,
        )
        .map_err(super::routed::construction_error)
    }
}
