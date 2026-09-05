use super::*;
pub(in crate::composition::mlx) fn bind_routed_text(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    selected: eredu_architectures::SelectedRoutedTextRealization,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error> {
    if selected.plan().relu2().is_some() {
        return eredu_architectures::visit_relu2_routed_text_architecture::<
            MlxNeuralBackend,
            MlxHybridState,
            _,
        >(
            inspection,
            selected,
            store,
            stream,
            Relu2RoutedBindingVisitor {
                stream,
                weights_stream,
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()));
    }
    let uses_pooling_attention = selected
        .text()
        .state()
        .components()
        .iter()
        .any(|component| {
            matches!(
                component.component().role(),
                eredu_core::cache::StateComponentRole::Fixed(
                    eredu_core::cache::StateTensorRole::Pooling { .. }
                )
            )
        });
    if uses_pooling_attention {
        return eredu_architectures::visit_pooling_routed_text_architecture::<
            MlxNeuralBackend,
            MlxPoolingAttentionState,
            _,
        >(
            inspection,
            selected,
            store,
            stream,
            PoolingRoutedBindingVisitor {
                stream,
                weights_stream,
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()));
    }
    eredu_architectures::visit_gated_routed_text_architecture::<
        MlxNeuralBackend,
        MlxHybridState,
        _,
    >(
        inspection,
        selected,
        store,
        stream,
        RoutedBindingVisitor {
            stream,
            weights_stream,
        },
    )
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

pub(super) fn selected_addressable_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<crate::backend::runtime::residency::parameter_bank::AddressableParameterBank, Error> {
    let selected =
        crate::backend::runtime::residency::parameter_bank::entries_from_selected_members(
            members,
            store.as_ref(),
        )?;
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank::new_selected_shared(
        store,
        selected,
        options,
        weights_stream.clone(),
        stream.clone(),
    )
    .map_err(Into::into)
}

pub(super) fn shard_addressable_members(
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
                    eredu_runtime::AddressableBankParameter::new(
                        parameter.binding_name(),
                        parameter.task().clone(),
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
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    layout: &eredu_runtime::LocalModelLayout,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<
    (
        std::collections::BTreeMap<eredu_runtime::ParameterBankKey, u64>,
        MlxSharedAddressableBank,
    ),
    Error,
> {
    let members = shard_addressable_members(members, store.as_ref(), layout)?;
    let bank = selected_addressable_bank(&members, store, options, weights_stream, stream)?;
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
pub(super) struct Relu2RoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
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
        prepared: eredu_architectures::PreparedRelu2RoutedTextArchitecture<A>,
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
        let capability_estimate = prepared.text().capability_estimate().clone();
        let effective_model_type = prepared.text().effective_model_type().to_owned();
        let selected_residency = prepared.text().selected().residency();
        let bank_residency = prepared.bank_residency();
        let mechanisms: MlxReplicatedTextMechanisms<A, MlxHybridState> =
            MlxReplicatedTextMechanisms::new(Arc::clone(&store), self.stream, self.weights_stream);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        match bank_residency {
            eredu_runtime::ParameterBankResidency::WithLayer => {
                let session = prepared
                    .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, self.stream)
                    .map_err(Error::ArchitectureModel)?;
                Ok(Box::new(CompletedReplicatedText::from_session(
                    session,
                    prompt_cache_identity,
                    capability_estimate,
                    effective_model_type,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )))
            }
            eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
                let bank = selected_addressable_bank(
                    prepared.addressable_members(),
                    store,
                    options,
                    self.weights_stream,
                    self.stream,
                )?;
                let session = prepared
                    .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                        mechanisms,
                        bank,
                        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                        self.stream,
                    )
                    .map_err(Error::ArchitectureModel)?;
                Ok(Box::new(CompletedReplicatedText::from_session(
                    session,
                    prompt_cache_identity,
                    capability_estimate,
                    effective_model_type,
                    selected_residency,
                    None,
                    None,
                    None,
                    true,
                    self.stream,
                )))
            }
            _ => Err(Error::ArchitectureModel(
                "unsupported selected addressable bank residency".into(),
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct RoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

#[derive(Clone, Copy)]
pub(super) struct PoolingRoutedBindingVisitor<'a> {
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

pub(super) fn bind_prepared_routed<A, S>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    S: MlxStateMechanisms + 'static,
    A: ReplicatedTextArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, S>
        + 'static,
    A::StaticModules: Clone,
    A::Error: std::fmt::Display,
{
    let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
    let capability_estimate = prepared.text().capability_estimate().clone();
    let effective_model_type = prepared.text().effective_model_type().to_owned();
    let selected_residency = prepared.text().selected().residency();
    let bank_residency = prepared.bank_residency();
    let mechanisms: MlxReplicatedTextMechanisms<A, S> =
        MlxReplicatedTextMechanisms::new(Arc::clone(&store), stream, weights_stream);
    #[cfg(test)]
    crate::tests::support::path_instrumentation::constructor();
    match bank_residency {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let session = prepared
                .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, stream)
                .map_err(Error::ArchitectureModel)?;
            Ok(Box::new(CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability_estimate,
                effective_model_type,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )))
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            let bank = selected_addressable_bank(
                prepared.addressable_members(),
                store,
                options,
                weights_stream,
                stream,
            )?;
            let session = prepared
                .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                    mechanisms,
                    bank,
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    stream,
                )
                .map_err(Error::ArchitectureModel)?;
            Ok(Box::new(CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability_estimate,
                effective_model_type,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )))
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported selected addressable bank residency".into(),
        )),
    }
}

pub(super) fn bind_prepared_routed_prediction<A, S, P>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    extension: P,
    selected: eredu_runtime::SelectedSpeculativeRealization,
    capability: eredu_architectures::capability::CapabilityEstimate,
    store: Arc<dyn CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
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
    let prompt_cache_identity = prepared.text().prompt_cache_identity().clone();
    let effective_model_type = prepared.text().effective_model_type().to_owned();
    let selected_residency = prepared.text().selected().residency();
    let bank_residency = prepared.bank_residency();
    let mechanisms =
        MlxReplicatedTextMechanisms::<A, S>::new(Arc::clone(&store), stream, weights_stream);
    let prediction = SelectedPrediction {
        extension,
        selected,
    };
    #[cfg(test)]
    crate::tests::support::path_instrumentation::constructor();
    match bank_residency {
        eredu_runtime::ParameterBankResidency::WithLayer => {
            let session = prepared
                .construct_resident_session::<MlxNeuralBackend, _>(mechanisms, stream)
                .map_err(Error::ArchitectureModel)?;
            CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability.clone(),
                effective_model_type,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )
            .with_prediction(prediction, capability)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
        }
        eredu_runtime::ParameterBankResidency::IndependentCache(options) => {
            let bank = selected_addressable_bank(
                prepared.addressable_members(),
                store,
                options,
                weights_stream,
                stream,
            )?;
            let session = prepared
                .construct_addressable_session::<MlxNeuralBackend, _, _, _>(
                    mechanisms,
                    bank,
                    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
                    stream,
                )
                .map_err(Error::ArchitectureModel)?;
            CompletedReplicatedText::from_session(
                session,
                prompt_cache_identity,
                capability.clone(),
                effective_model_type,
                selected_residency,
                None,
                None,
                None,
                true,
                stream,
            )
            .with_prediction(prediction, capability)
            .map(|model| Box::new(model) as Box<dyn ErasedReplicatedTextExecutable>)
        }
        _ => Err(Error::ArchitectureModel(
            "unsupported selected addressable bank residency".into(),
        )),
    }
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
        store: Arc<dyn CheckpointSource>,
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
        )
    }
}

impl
    eredu_architectures::GatedRoutedTextArchitectureVisitor<
        MlxNeuralBackend,
        MlxPoolingAttentionState,
    > for PoolingRoutedBindingVisitor<'_>
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
        store: Arc<dyn CheckpointSource>,
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
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream)
    }
}

impl eredu_architectures::GatedRoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
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
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        bind_prepared_routed(prepared, store, self.stream, self.weights_stream)
    }
}
