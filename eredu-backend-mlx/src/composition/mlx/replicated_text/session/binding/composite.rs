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
    let banks = session.execution_strategy().parameter_banks();
    let (text, processor, admission) = facts.into_parts();
    let (identity, capability, model_type, residency) = text.into_parts();
    finalizer.finish(
        CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
            session, admission, processor, identity, capability, model_type, residency, None, None,
            None, true, stream,
        )
        .with_parameter_banks(banks)?,
    )
}

/// Family-agnostic MLX binder for architecture-owned composite ingress.
#[derive(Clone, Copy)]
pub(crate) struct CompositeBindingVisitor<'a> {
    pub stream: &'a Stream,
    pub weights_stream: &'a Stream,
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
    fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, MlxTensor>,
        active: bool,
    ) {
        super::super::prediction::parameters::publish::<PreparedCompositeArchitecture<A>, P>(
            &mut self.extension,
            values,
            active,
        );
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
    ) -> Result<bool, Error> {
        super::super::prediction::parameters::with_slots::<PreparedCompositeArchitecture<A>, P>(
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
        model: &mut CompletedComposite<A, D, Self>,
        continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        let selected = &model.prediction.selected;
        let input = MlxCompositePredictionInput::<A> {
            admission: &model.admission,
            processor: &model.processor,
        };
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
        store: Arc<dyn CheckpointSource>,
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
        CompletedComposite::new_with_residency(
            prepared,
            store,
            self.stream,
            self.weights_stream,
            residency,
        )?
        .with_prediction(prediction, self.capability)
        .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        extension: <PreparedCompositeArchitecture<A> as eredu_architectures::prediction_extension::MaterializedPredictionTarget<MlxNeuralBackend>>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: Arc<dyn CheckpointSource>,
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
        >::new(Arc::clone(&store), self.stream, self.weights_stream);
        let mut prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        mechanisms.set_prediction_residency(super::super::prediction::parameters::residency::<
            PreparedCompositeArchitecture<A>,
            _,
        >(&mut prediction.extension)?);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.stream,
            |banks, residency, rows| {
                super::routed::selected_parameter_providers(
                    banks,
                    Arc::clone(&store),
                    residency,
                    rows,
                    self.weights_stream,
                    self.stream,
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

    fn visit_prediction<A, W>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        prediction: W,
        target_source: Arc<dyn CheckpointSource>,
        provider_source: Arc<dyn CheckpointSource>,
        binding: eredu_architectures::prepared_execution::PredictionBinding,
    ) -> Result<
        Self::Output,
        eredu_architectures::prepared_execution::PreparedExecutionError<Self::Error>,
    >
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        W: eredu_architectures::prediction_extension::PreparedRoutedPrediction<
            MlxNeuralBackend,
            PreparedCompositeArchitecture<A>,
        >,
    {
        bind_retained_composite_prediction(
            prepared,
            prediction,
            target_source,
            provider_source,
            binding,
            self.stream,
            self.weights_stream,
        )
        .map_err(eredu_architectures::prepared_execution::PreparedExecutionError::Backend)
    }

    fn visit<A>(
        self,
        prepared: PreparedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::Error: std::fmt::Display,
    {
        CompletedComposite::new(prepared, store, self.stream, self.weights_stream)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
    }

    fn visit_routed<A>(
        self,
        prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mechanisms: MlxReplicatedTextMechanisms<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        > = MlxReplicatedTextMechanisms::new(Arc::clone(&store), self.stream, self.weights_stream);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_composite_session(
            prepared,
            mechanisms,
            self.stream,
            |banks, residency, rows| {
                super::routed::selected_parameter_providers(
                    banks,
                    Arc::clone(&store),
                    residency,
                    rows,
                    self.weights_stream,
                    self.stream,
                )
            },
            (self.stream, OrdinaryReplicatedFinalizer),
            finish_routed_composite_session,
            finish_routed_composite_session,
        )
        .map_err(super::routed::construction_error)
    }
}

/// Materializes the retained prediction source and moves its banks from the
/// composite target's shared provider into the prediction extension.
fn bind_retained_composite_prediction<A, W>(
    prepared: PreparedRoutedCompositeTextArchitecture<A, A::AdmissionConfig>,
    prediction: W,
    target_source: Arc<dyn CheckpointSource>,
    provider_source: Arc<dyn CheckpointSource>,
    binding: eredu_architectures::prepared_execution::PredictionBinding,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::InputPartPlan: 'static,
    A::StaticModules: Clone,
    W: eredu_architectures::prediction_extension::PreparedRoutedPrediction<
        MlxNeuralBackend,
        PreparedCompositeArchitecture<A>,
    >,
{
    use crate::composition::mlx::replicated_text::prediction::MlxPredictionMaterializationContext;
    let auxiliary = prediction.banks();
    let mut context = MlxPredictionMaterializationContext::new(
        prediction.source().clone(),
        stream,
        weights_stream,
    );
    let mut extension = prediction
        .materialize::<MlxEmbeddedPredictionMaterializer>(&mut context, |_, selected| {
            MlxHybridState::realize(selected, None, 0)
        })
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let mut mechanisms = MlxReplicatedTextMechanisms::<
        PreparedCompositeArchitecture<A>,
        MlxHybridState,
    >::new(target_source, stream, weights_stream);
    mechanisms.set_prediction_residency(super::super::super::prediction::parameters::residency::<
        PreparedCompositeArchitecture<A>,
        _,
    >(&mut extension)?);
    let (prepared, processor, admission) = prepared.into_parts();
    let facts = eredu_architectures::prepared_execution::PreparedCompositeSessionFacts::new(
        eredu_architectures::prepared_execution::PreparedTextSessionFacts::from_prepared(
            prepared.text(),
        ),
        processor,
        admission,
    );
    let residency = prepared.bank_residency();
    let (banks, rows) = super::routed::selected_parameter_providers(
        prepared.banks(),
        provider_source,
        residency,
        prepared.row_lookups(),
        weights_stream,
        stream,
    )?;
    macro_rules! finish {
        ($result:expr) => {{
            let (session, provider) = $result.map_err(Error::ArchitectureModel)?;
            let provider = provider.ok_or_else(|| {
                Error::ArchitectureModel(
                    "prediction banks were not moved from composite target construction".into(),
                )
            })?;
            let extension =
                W::with_provider::<MlxEmbeddedPredictionMaterializer, _>(extension, provider);
            finish_routed_composite_session(
                (
                    stream,
                    PredictionReplicatedFinalizer {
                        prediction: SelectedPrediction {
                            extension,
                            selected: binding.selected().clone(),
                        },
                        capability: binding.capability().clone(),
                    },
                ),
                session,
                facts,
            )
        }};
    }
    match residency {
        eredu_runtime::ParameterBankResidency::WithLayer => finish!(prepared
            .construct_resident_session::<MlxNeuralBackend, _, _>(
                mechanisms, rows, &auxiliary, stream
            )),
        eredu_runtime::ParameterBankResidency::IndependentCache(_) => finish!(prepared
            .construct_addressable_session::<MlxNeuralBackend, _, _, _, _>(
            mechanisms, banks, rows, &auxiliary, stream
        )),
        _ => Err(Error::ArchitectureModel(
            "selected prediction bank residency has no binding".into(),
        )),
    }
}
