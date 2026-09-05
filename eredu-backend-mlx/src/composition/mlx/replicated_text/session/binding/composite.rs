use super::*;

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
        CompletedComposite::new(prepared, store, self.stream, self.weights_stream)?
            .with_prediction(
                SelectedPrediction {
                    extension,
                    selected: self.selected,
                },
                self.capability,
            )
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
        let prompt_cache_identity = prepared.routed().text().prompt_cache_identity().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let selected_residency = prepared.routed().text().selected().residency();
        let bank_residency = prepared.routed().bank_residency();
        let (routed, processor, admission) = prepared.into_parts();
        let mechanisms = MlxReplicatedTextMechanisms::<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        >::new(Arc::clone(&store), self.stream, self.weights_stream);
        let prediction = SelectedPrediction {
            extension,
            selected: self.selected,
        };
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = routed
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                    session,
                    admission,
                    processor,
                    prompt_cache_identity,
                    self.capability.clone(),
                    effective_model_type,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )
                .with_prediction(prediction, self.capability)
                .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    routed.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = routed
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                    session,
                    admission,
                    processor,
                    prompt_cache_identity,
                    self.capability.clone(),
                    effective_model_type,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )
                .with_prediction(prediction, self.capability)
                .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected composite bank residency".into(),
            )),
        }
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
        let prompt_cache_identity = prepared.routed().text().prompt_cache_identity().clone();
        let capability_estimate = prepared.capability_estimate().clone();
        let effective_model_type = prepared.effective_model_type().to_owned();
        let selected_residency = prepared.routed().text().selected().residency();
        let bank_residency = prepared.routed().bank_residency();
        let (routed, processor, admission) = prepared.into_parts();
        let mechanisms: MlxReplicatedTextMechanisms<
            PreparedCompositeArchitecture<A>,
            MlxHybridState,
        > = MlxReplicatedTextMechanisms::new(Arc::clone(&store), self.stream, self.weights_stream);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = routed
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                Ok(Box::new(
                    CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                        session,
                        admission,
                        processor,
                        prompt_cache_identity,
                        capability_estimate,
                        effective_model_type,
                        selected_residency,
                        None,
                        None,
                        None,
                        true,
                        self.stream,
                    ),
                ))
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    routed.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = routed
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                Ok(Box::new(
                    CompletedComposite::<A, _, NoSelectedPrediction>::from_session(
                        session,
                        admission,
                        processor,
                        prompt_cache_identity,
                        capability_estimate,
                        effective_model_type,
                        selected_residency,
                        None,
                        None,
                        None,
                        true,
                        self.stream,
                    ),
                ))
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected composite bank residency".into(),
            )),
        }
    }
}
