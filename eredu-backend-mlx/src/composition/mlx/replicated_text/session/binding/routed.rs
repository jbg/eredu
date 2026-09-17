use super::*;

pub(super) fn construction_error(
    error: eredu_architectures::prepared_execution::PreparedExecutionError<Error>,
) -> Error {
    match error {
        eredu_architectures::prepared_execution::PreparedExecutionError::Backend(error) => error,
        error => Error::ArchitectureModel(error.to_string()),
    }
}

fn finish_routed_session<A, S, D, F>(
    (stream, finalizer): (&Stream, F),
    session: ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
    facts: eredu_architectures::prepared_execution::PreparedTextSessionFacts,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
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
    F: ReplicatedExecutableFinalizer<A, S>,
{
    let (identity, capability, model_type, residency) = facts.into_parts();
    let banks = session.execution_strategy().parameter_banks()?;
    finalizer.finish(
        CompletedReplicatedText::from_session(
            session, identity, capability, model_type, residency, None, None, None, true, stream,
        )
        .with_parameter_banks(banks)?,
    )
}

pub(super) fn selected_addressable_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    options: eredu_runtime::ParameterBankLoadOptions,
    weights_stream: &Stream,
    stream: &Stream,
    addressable_manager: Option<&AddressableManagerSlot>,
) -> Result<crate::backend::runtime::residency::parameter_bank::AddressableParameterBank, Error> {
    let selected =
        crate::backend::runtime::residency::parameter_bank::entries_from_selected_members(
            members,
            store.as_ref(),
        )?;
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank::new_selected_shared_with_manager(
        store,
        selected,
        options,
        weights_stream.clone(),
        stream.clone(),
            addressable_manager.and_then(std::cell::Cell::take),
        )
    .map_err(Into::into)
}

/// Binds all selected banks to one cache and its single residency budget.
pub(super) fn selected_addressable_banks(
    banks: &std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        eredu_architectures::routed_text::SelectedRoutedBank,
    >,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    options: eredu_runtime::ParameterBankLoadOptions,
    weights_stream: &Stream,
    stream: &Stream,
    addressable_manager: Option<&AddressableManagerSlot>,
) -> Result<
    std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        (
            MlxSharedAddressableBank,
            crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
        ),
    >,
    Error,
> {
    let members = banks
        .iter()
        .flat_map(|(_, bank)| bank.addressable_members().iter().cloned())
        .collect::<Vec<_>>();
    let pool = MlxSharedAddressableBank::new(selected_addressable_bank(
        &members,
        store,
        options,
        weights_stream,
        stream,
            addressable_manager,
        )?);
    banks
        .keys()
        .map(|id| {
            let bank = pool.scoped(id.value() as usize)?;
            let movement = crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement::for_bank(
                bank.clone(), options);
            Ok((*id, (bank, movement)))
        })
        .collect()
}

pub(crate) fn shard_addressable_members(
    members: &[eredu_runtime::AddressableBankMember],
    store: &dyn CheckpointSource,
    layout: &eredu_runtime::LocalModelLayout,
) -> Result<Vec<eredu_runtime::AddressableBankMember>, Error> {
    members
        .iter()
        .map(|member| {
            let source_bindings = member
                .parameters()
                .iter()
                .map(|parameter| {
                    eredu_runtime::WeightBinding::from_recipe(
                        parameter.binding_name(),
                        parameter.recipe().clone(),
                        parameter.source_bytes(),
                    )?
                    .with_logical_target(parameter.task().name())
                    .map_err(Into::into)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let bindings = shard_addressable_member_bindings(source_bindings, store, layout)?;
            let parameters = member
                .parameters()
                .iter()
                .zip(bindings)
                .map(|(parameter, binding)| {
                    let recipe = binding.source_recipe();
                    let metadata = recipe.infer(store)?;
                    let selected_bytes = eredu_runtime::selected_addressable_parameter_bytes(
                        parameter.task(),
                        &metadata,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
                    let companions = parameter.quantization_companions().cloned();
                    eredu_runtime::AddressableBankParameter::from_shared_task(
                        parameter.binding_name(),
                        parameter.shared_task().clone(),
                        recipe,
                        metadata,
                        selected_bytes,
                        companions,
                    )
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            eredu_runtime::AddressableBankMember::new(
                member.key(),
                member.placement().clone(),
                parameters,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn selected_addressable_partition_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    options: eredu_runtime::ParameterBankLoadOptions,
    layout: &eredu_runtime::LocalModelLayout,
    weights_stream: &Stream,
    stream: &Stream,
    addressable_manager: Option<&AddressableManagerSlot>,
) -> Result<
    (
        std::collections::BTreeMap<eredu_runtime::ParameterBankKey, u64>,
        MlxSharedAddressableBank,
    ),
    Error,
> {
    let members = shard_addressable_members(members, store.as_ref(), layout)?;
    let bank = selected_addressable_bank(&members, store, options, weights_stream, stream,
            addressable_manager,
        )?;
    let selected_member_bytes = members
        .iter()
        .map(|member| {
            let bytes = <crate::backend::runtime::residency::parameter_bank::AddressableParameterBank as eredu_runtime::AddressableGroupedBank<MlxNeuralBackend>>::member_bytes(
                &bank,
                member.key(),
            )
            .expect("constructed addressable bank retains every selected member");
            (member.key(), bytes)
        })
        .collect();
    Ok((selected_member_bytes, MlxSharedAddressableBank::new(bank)))
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct Relu2RoutedBindingVisitor<'a> {
    pub(in crate::composition::mlx) addressable_manager: Option<&'a AddressableManagerSlot>,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
    pub(in crate::composition::mlx) layerwise_manager: Option<&'a LayerwiseManagerSlot>,
}

impl eredu_architectures::Relu2RoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for Relu2RoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mut mechanisms: MlxReplicatedTextMechanisms<A, MlxHybridState> =
            MlxReplicatedTextMechanisms::new(store.clone(), self.stream, self.weights_stream)?;
        mechanisms.set_prepared_layerwise_manager(
            self.layerwise_manager.and_then(std::cell::Cell::take),
        );
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_session(
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
            finish_routed_session,
            finish_routed_session,
        )
        .map_err(construction_error)
    }
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct RoutedBindingVisitor<'a> {
    pub(in crate::composition::mlx) addressable_manager: Option<&'a AddressableManagerSlot>,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
    pub(in crate::composition::mlx) layerwise_manager: Option<&'a LayerwiseManagerSlot>,
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct PoolingRoutedBindingVisitor<'a> {
    pub(in crate::composition::mlx) addressable_manager: Option<&'a AddressableManagerSlot>,
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
    pub(in crate::composition::mlx) layerwise_manager: Option<&'a LayerwiseManagerSlot>,
}

pub(super) fn bind_prepared_routed<A, S>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
    layerwise_manager: Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>,
    addressable_manager: Option<&AddressableManagerSlot>,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let mut mechanisms: MlxReplicatedTextMechanisms<A, S> =
        MlxReplicatedTextMechanisms::new(store.clone(), stream, weights_stream)?;
    mechanisms.set_prepared_layerwise_manager(layerwise_manager);
    #[cfg(test)]
    crate::tests::support::path_instrumentation::constructor();
    eredu_architectures::prepared_execution::construct_selected_routed_session(
        prepared,
        mechanisms,
        stream,
        |banks, options| {
            super::routed::selected_addressable_banks(
                banks,
                store.clone(),
                options,
                weights_stream,
                stream,
            addressable_manager,
        )
        },
        (stream, OrdinaryReplicatedFinalizer),
        finish_routed_session,
        finish_routed_session,
    )
    .map_err(construction_error)
}

pub(super) fn bind_prepared_routed_prediction<A, S, P>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    extension: P,
    selected: eredu_runtime::SelectedSpeculativeRealization,
    capability: eredu_architectures::capability::CapabilityEstimate,
    store: eredu_checkpoint::store::RetainedCheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
    layerwise_manager: Option<crate::backend::runtime::execution::generic::PreparedLayerwiseManager>,
    addressable_manager: Option<&AddressableManagerSlot>,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
    P: eredu_architectures::prediction_extension::MaterializedPredictionExecutor<
            A,
            MlxNeuralBackend,
            MlxEmbeddedPredictionMaterializer,
        > + 'static,
{
    let mut mechanisms =
        MlxReplicatedTextMechanisms::<A, S>::new(store.clone(), stream, weights_stream)?;
    mechanisms.set_prepared_layerwise_manager(layerwise_manager);
    let mut prediction = SelectedPrediction {
        extension,
        selected,
    };
    mechanisms.set_prediction_residency(super::super::prediction::parameters::residency::<A, P>(
        &mut prediction.extension,
    )?);
    #[cfg(test)]
    crate::tests::support::path_instrumentation::constructor();
    eredu_architectures::prepared_execution::construct_selected_routed_session(
        prepared,
        mechanisms,
        stream,
        |banks, options| {
            super::routed::selected_addressable_banks(
                banks,
                store.clone(),
                options,
                weights_stream,
                stream,
            addressable_manager,
        )
        },
        (
            stream,
            PredictionReplicatedFinalizer {
                prediction,
                capability,
            },
        ),
        finish_routed_session,
        finish_routed_session,
    )
    .map_err(construction_error)
}

impl<S>
    eredu_architectures::routed_text::RoutedPredictionTargetVisitor<
        MlxNeuralBackend,
        S,
        MlxEmbeddedPredictionMaterializer,
    > for PredictionBindingVisitor<'_>
where
    S: MlxStateMechanisms + 'static,
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            MlxNeuralBackend,
        >>::Extension<MlxEmbeddedPredictionMaterializer>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                MlxNeuralBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed_prediction(
            prepared,
            extension,
            self.selected,
            self.capability,
            store,
            self.stream,
            self.weights_stream,
            self.layerwise_manager.and_then(std::cell::Cell::take),
            self.addressable_manager,
        )
    }
}

impl eredu_architectures::RoutedTextArchitectureVisitor<MlxNeuralBackend, MlxPoolingAttentionState>
    for PoolingRoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<
                MlxNeuralBackend,
                MlxPoolingAttentionState,
                Error = eredu_nn::Error,
            > + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxPoolingAttentionState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream,
            self.layerwise_manager.and_then(std::cell::Cell::take),
            self.addressable_manager,
        )
    }
}

impl eredu_architectures::RoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for RoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        store: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream,
            self.layerwise_manager.and_then(std::cell::Cell::take),
            self.addressable_manager,
        )
    }
}
