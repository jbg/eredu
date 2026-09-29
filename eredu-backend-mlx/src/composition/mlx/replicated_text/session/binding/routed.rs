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
    let banks = session.execution_strategy().parameter_banks();
    finalizer.finish(
        CompletedReplicatedText::from_session(
            session, identity, capability, model_type, residency, None, None, None, true, stream,
        )
        .with_parameter_banks(banks)?,
    )
}

pub(super) fn selected_addressable_bank(
    members: &[eredu_runtime::AddressableBankMember],
    store: Arc<dyn CheckpointSource>,
    rows: &eredu_runtime::SelectedRowLookups,
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
        rows,
        weights_stream.clone(),
        stream.clone(),
    )
    .map_err(Into::into)
}

/// Binds grouped and row providers against one selected physical cache.
pub(super) fn selected_parameter_providers(
    banks: &std::collections::BTreeMap<
        eredu_runtime::RoutedBankId,
        eredu_architectures::routed_text::SelectedRoutedBank,
    >,
    store: Arc<dyn CheckpointSource>,
    residency: eredu_runtime::ParameterBankResidency,
    rows: Option<&eredu_runtime::SelectedRowLookups>,
    weights_stream: &Stream,
    stream: &Stream,
) -> Result<
    (
        std::collections::BTreeMap<
            eredu_runtime::RoutedBankId,
            (
                MlxSharedAddressableBank,
                crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
            ),
        >,
        Option<crate::backend::runtime::residency::parameter_bank::MlxRowLookups>,
    ),
    Error,
> {
    use crate::backend::runtime::residency::parameter_bank::{
        MlxIndexedMovement, MlxRowLookupSupport, MlxRowLookups,
    };
    if let (Some(rows), eredu_runtime::ParameterBankResidency::IndependentCache(options)) =
        (rows, residency)
    {
        if rows.options() != options {
            return Err(eredu_runtime::RowLookupSelectionError::PoolMismatch.into());
        }
    }
    let addressable = matches!(
        residency,
        eredu_runtime::ParameterBankResidency::IndependentCache(_)
    );
    let empty;
    let selected = if let Some(rows) = rows {
        rows
    } else {
        let eredu_runtime::ParameterBankResidency::IndependentCache(options) = residency else {
            return Err(Error::ArchitectureModel(
                "provider binding has neither addressable banks nor rows".into(),
            ));
        };
        empty = eredu_runtime::SelectedRowLookups::select(
            Default::default(),
            options,
            0,
            &MlxRowLookupSupport,
        )?;
        &empty
    };
    let members = if addressable {
        banks
            .values()
            .flat_map(|bank| bank.addressable_members().iter().cloned())
            .collect::<Vec<_>>()
    } else {
        vec![]
    };
    let pool = selected_addressable_bank(&members, store, selected, weights_stream, stream)?;
    let pool = MlxSharedAddressableBank::new(pool);
    let row_provider = rows
        .map(|_| MlxRowLookups::bind_shared(&pool, selected, weights_stream, stream))
        .transpose()?;
    let grouped = if addressable {
        banks
            .keys()
            .map(|id| Ok((*id, (pool.scoped(id.value() as usize)?, MlxIndexedMovement))))
            .collect::<Result<_, Error>>()?
    } else {
        Default::default()
    };
    Ok((grouped, row_provider))
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
    store: Arc<dyn CheckpointSource>,
    options: eredu_runtime::ParameterBankLoadOptions,
    layout: &eredu_runtime::LocalModelLayout,
    selected_rows: Option<&eredu_runtime::SelectedRowLookups>,
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
    let empty_rows = eredu_runtime::SelectedRowLookups::select(
        Default::default(),
        options,
        0,
        &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
    )?;
    let rows = selected_rows.unwrap_or(&empty_rows);
    if rows.options() != options {
        return Err(eredu_runtime::RowLookupSelectionError::PoolMismatch.into());
    }
    let bank = selected_addressable_bank(&members, store, rows, weights_stream, stream)?;
    let selected_member_bytes = members
        .iter()
        .map(|member| {
            let bytes = <crate::backend::runtime::residency::parameter_bank::AddressableParameterBank as eredu_runtime::ParameterBank<MlxNeuralBackend>>::member_bytes(
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
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
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
        store: Arc<dyn CheckpointSource>,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mechanisms: MlxReplicatedTextMechanisms<A, MlxHybridState> =
            MlxReplicatedTextMechanisms::new(Arc::clone(&store), self.stream, self.weights_stream);
        #[cfg(test)]
        crate::tests::support::path_instrumentation::constructor();
        eredu_architectures::prepared_execution::construct_selected_routed_session(
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
            finish_routed_session,
            finish_routed_session,
        )
        .map_err(construction_error)
    }
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct RoutedBindingVisitor<'a> {
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
}

#[derive(Clone, Copy)]
pub(in crate::composition::mlx) struct PoolingRoutedBindingVisitor<'a> {
    pub(in crate::composition::mlx) stream: &'a Stream,
    pub(in crate::composition::mlx) weights_stream: &'a Stream,
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
    let mechanisms: MlxReplicatedTextMechanisms<A, S> =
        MlxReplicatedTextMechanisms::new(Arc::clone(&store), stream, weights_stream);
    #[cfg(test)]
    crate::tests::support::path_instrumentation::constructor();
    eredu_architectures::prepared_execution::construct_selected_routed_session(
        prepared,
        mechanisms,
        stream,
        |banks, residency, rows| {
            super::routed::selected_parameter_providers(
                banks,
                Arc::clone(&store),
                residency,
                rows,
                weights_stream,
                stream,
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
    let mut mechanisms =
        MlxReplicatedTextMechanisms::<A, S>::new(Arc::clone(&store), stream, weights_stream);
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
        |banks, residency, rows| {
            super::routed::selected_parameter_providers(
                banks,
                Arc::clone(&store),
                residency,
                rows,
                weights_stream,
                stream,
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

impl eredu_architectures::RoutedTextArchitectureVisitor<MlxNeuralBackend, MlxHybridState>
    for RoutedBindingVisitor<'_>
{
    type Output = Box<dyn ErasedReplicatedTextExecutable>;
    type Error = Error;

    fn construction_started(&mut self) {
        #[cfg(test)]
        crate::tests::support::path_instrumentation::architecture_construction();
    }

    fn visit_prediction<A, W>(
        self,
        prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
        prediction: W,
        target_source: Arc<dyn CheckpointSource>,
        provider_source: Arc<dyn CheckpointSource>,
        binding: eredu_architectures::prepared_execution::PredictionBinding,
    ) -> Result<Self::Output, eredu_architectures::routed_text::RoutedTextDispatchError<Self::Error>>
    where
        A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
            + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
            + 'static,
        A::StaticModules: Clone,
        W: eredu_architectures::prediction_extension::PreparedRoutedPrediction<MlxNeuralBackend, A>,
    {
        bind_retained_routed_prediction(
            prepared,
            prediction,
            target_source,
            provider_source,
            binding,
            self.stream,
            self.weights_stream,
        )
        .map_err(eredu_architectures::routed_text::RoutedTextDispatchError::Backend)
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

/// Realizes architecture-declared prediction owners, selected state and moved banks.
fn bind_retained_routed_prediction<A, W>(
    prepared: eredu_architectures::PreparedRoutedTextArchitecture<A>,
    prediction: W,
    target_source: Arc<dyn CheckpointSource>,
    provider_source: Arc<dyn CheckpointSource>,
    binding: eredu_architectures::prepared_execution::PredictionBinding,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn ErasedReplicatedTextExecutable>, Error>
where
    A: ReplicatedTextArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error>
        + eredu_runtime::RoutedLayeredArchitecture<MlxNeuralBackend, MlxHybridState>
        + 'static,
    A::StaticModules: Clone,
    W: eredu_architectures::prediction_extension::PreparedRoutedPrediction<MlxNeuralBackend, A>,
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
    let mut mechanisms = MlxReplicatedTextMechanisms::<A, MlxHybridState>::new(
        target_source,
        stream,
        weights_stream,
    );
    mechanisms.set_prediction_residency(super::super::super::prediction::parameters::residency::<
        A,
        _,
    >(&mut extension)?);
    let facts = eredu_architectures::prepared_execution::PreparedTextSessionFacts::from_prepared(
        prepared.text(),
    );
    let residency = prepared.bank_residency();
    let (banks, rows) = selected_parameter_providers(
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
                    "prediction banks were not moved from target construction".into(),
                )
            })?;
            let extension =
                W::with_provider::<MlxEmbeddedPredictionMaterializer, _>(extension, provider);
            finish_routed_session(
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

#[cfg(test)]
mod row_provider_tests {
    use super::*;
    use eredu_runtime::{ParameterBankAccess, RowLookupProvider};
    #[test]
    fn resident_grouped_execution_binds_a_bounded_row_only_pool() {
        use eredu_checkpoint::{
            recipe::DerivedWeightRecipe,
            rows::PreparedRowSource,
            store::{MemoryWeightStore, TensorSelection},
        };
        use eredu_core::residency::{
            OffloadConfig, OffloadUnitId, OffloadUnitRange, ResidencyPolicy,
        };
        let source: Arc<dyn CheckpointSource> = Arc::new(
            MemoryWeightStore::from_safetensors([(
                "table".into(),
                safetensors::Dtype::F32,
                vec![4, 2],
                (1..=8).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            )])
            .unwrap(),
        );
        let spec = eredu_runtime::RowLookupSpec {
            parameter: eredu_nn::ParameterId::new("rows").unwrap(),
            bank: 1,
            unit: 0,
            rows: 4,
            dimensions: 2,
            encoding: eredu_runtime::RowEncoding::Dense,
            output_type: eredu_nn::TensorElementType::F32,
        };
        let table = PreparedRowSource::new(
            source.clone(),
            DerivedWeightRecipe::source("table", TensorSelection::Full),
        )
        .unwrap();
        let range = eredu_runtime::RowResidencyRange::new(
            OffloadUnitRange::new(
                OffloadUnitId::new("rows").unwrap(),
                0,
                4,
                8,
                ResidencyPolicy::Cacheable,
            )
            .unwrap(),
            table,
            "value",
        )
        .unwrap();
        let entry = eredu_runtime::PreparedRowLookup::new(
            range,
            spec.clone(),
            None,
            eredu_runtime::RowLookupLimits {
                requests: 3,
                rows_per_acquisition: 2,
                acquisition_bytes: 16,
                host_bytes: 384,
                output_bytes: 72,
            },
        )
        .unwrap();
        let options = eredu_runtime::ParameterBankLoadOptions::new(
            OffloadConfig::new(Some(16), Some(64), 1).unwrap(),
            1024,
            1024,
        )
        .unwrap();
        let selected = eredu_runtime::SelectedRowLookups::select(
            eredu_runtime::PreparedRowLookups::new([entry], 1).unwrap(),
            options,
            0,
            &crate::backend::runtime::residency::parameter_bank::MlxRowLookupSupport,
        )
        .unwrap();
        let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let (banks, rows) = selected_parameter_providers(
            &Default::default(),
            source.clone(),
            eredu_runtime::ParameterBankResidency::WithLayer,
            Some(&selected),
            &stream,
            &stream,
        )
        .unwrap();
        assert!(banks.is_empty());
        let mut rows = rows.unwrap();
        let result = rows
            .lookup_rows(&spec, &[3, 0, 3], ParameterBankAccess::Bulk, &stream)
            .unwrap();
        assert_eq!(
            result.as_array().evaluated().unwrap().as_slice::<f32>(),
            &[7., 8., 1., 2., 7., 8.]
        );
        let report =
            crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport::new(
                Default::default(),
            )
            .with_rows(rows.pool_report().unwrap().unwrap());
        assert_eq!(report.device_resident_bytes(), 16);
        assert_eq!(report.peak_device_resident_bytes(), 16);
        assert_eq!(
            report.rows().unwrap().requirements(),
            selected.requirements()
        );
        assert!(selected_parameter_providers(
            &Default::default(),
            source,
            eredu_runtime::ParameterBankResidency::IndependentCache(
                eredu_runtime::ParameterBankLoadOptions::default()
            ),
            Some(&selected),
            &stream,
            &stream
        )
        .is_err());
    }
}
