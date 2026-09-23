use super::*;
use eredu_checkpoint::store::SharedCheckpointSource;

pub(super) mod parameters;
pub(super) use parameters::MlxPredictionModule;

#[cfg(test)]
pub(super) type StatePresenceSnapshot = Vec<(i32, Vec<(eredu_core::cache::StateTensorRole, bool)>)>;

#[cfg(test)]
pub(super) type FixedNumericStateSnapshot = Vec<(
    usize,
    eredu_core::cache::StateTensorRole,
    Vec<i32>,
    Vec<f32>,
)>;

#[cfg(test)]
pub(super) type RetainedNumericStateSnapshot = Vec<(Vec<i32>, Vec<f32>)>;

#[cfg(test)]
pub(super) type CheckpointRestoreProbe = (
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    StatePresenceSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    FixedNumericStateSnapshot,
    Vec<f32>,
);

pub(super) trait MlxParameterBankTelemetry {
    fn parameter_banks(
        &self,
    ) -> std::collections::BTreeMap<eredu_runtime::RoutedBankId, MlxSharedAddressableBank> {
        Default::default()
    }
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    >;
}

impl MlxParameterBankTelemetry for eredu_runtime::DirectReplicatedTextExecution {
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<
        eredu_runtime::RoutedBankProviders<eredu_architectures::routed_text::PlannedResidentBank>,
    >
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl<E, G, R, I, T, U, V> MlxParameterBankTelemetry
    for eredu_runtime::PartitionedTextExecution<E, G, R, I, T, U, V>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

pub(super) type MlxAddressableBanks = eredu_runtime::RoutedBankProviders<
    eredu_architectures::routed_text::PlannedAddressableBank<
        MlxNeuralBackend,
        crate::backend::runtime::residency::parameter_bank::SharedAddressableParameterBank,
        crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
    >,
>;
impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<MlxAddressableBanks>
{
    fn parameter_banks(
        &self,
    ) -> std::collections::BTreeMap<eredu_runtime::RoutedBankId, MlxSharedAddressableBank> {
        self.provider()
            .banks()
            .iter()
            .map(|(id, provider)| (*id, provider.bank_storage().clone()))
            .collect()
    }
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        let banks = self
            .provider()
            .banks()
            .iter()
            .map(|(id, provider)| {
                provider
                    .bank_report()
                    .map(|report| (*id, report))
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            })
            .collect::<Result<_, _>>()?;
        Ok(Some(
            crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport::new(
                banks,
            ),
        ))
    }
}

pub(super) trait ErasedPredictionTargetState: std::any::Any {
    fn control_growth(&self, additional: u64) -> Option<u64>;
    fn control_estimate(&self) -> Option<eredu_core::execution_control::SnapshotEstimate>;
    fn control_copy(
        &self,
        stream: &Stream,
    ) -> Result<Box<dyn ErasedPredictionTargetState>, Exception>;

    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception>;
    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception>;
    fn as_any(&self) -> &dyn std::any::Any;
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
    fn offset(&self) -> i32;
}

impl<S> ErasedPredictionTargetState for S
where
    S: MlxStateMechanisms + 'static,
{
    fn control_growth(&self, additional: u64) -> Option<u64> {
        self.isolated_snapshot_growth(additional)
    }
    fn control_estimate(&self) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.isolated_snapshot_estimate()
    }
    fn control_copy(
        &self,
        stream: &Stream,
    ) -> Result<Box<dyn ErasedPredictionTargetState>, Exception> {
        self.isolated_snapshot(stream)
            .map(|state| Box::new(state) as Box<dyn ErasedPredictionTargetState>)
    }
    fn deep_clone_box(&self) -> Result<Box<dyn ErasedPredictionTargetState>, Exception> {
        self.deep_checkpoint()
            .map(|state| Box::new(state) as Box<dyn ErasedPredictionTargetState>)
    }

    fn restore_box(
        &mut self,
        checkpoint: &dyn ErasedPredictionTargetState,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let checkpoint = checkpoint
            .as_any()
            .downcast_ref::<S>()
            .ok_or_else(|| Exception::custom("prediction target checkpoint state type changed"))?;
        self.restore_checkpoint(checkpoint, stream)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }

    fn offset(&self) -> i32 {
        MlxStateMechanisms::offset(self)
    }
}

/// Opaque MLX storage for one ordinary target lane.
///
/// Neutral prediction membership and transaction metadata live in
/// `EmbeddedPredictionCache`; this wrapper supplies only native clone,
/// restore, type transfer, and frontier inspection mechanisms.
pub(crate) struct MlxPredictionTargetState(Option<Box<dyn ErasedPredictionTargetState>>);

impl MlxPredictionTargetState {
    pub(crate) fn control_growth(&self, additional: u64) -> Option<u64> {
        self.0.as_ref()?.control_growth(additional)
    }
    pub(crate) fn control_estimate(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        self.0.as_ref()?.control_estimate()
    }
    pub(crate) fn control_copy(&self, stream: &Stream) -> Result<Self, Exception> {
        self.0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target state is active"))?
            .control_copy(stream)
            .map(|state| Self(Some(state)))
    }

    pub(crate) fn new<S: MlxStateMechanisms + 'static>(state: S) -> Self {
        Self(Some(Box::new(state)))
    }

    pub(super) fn is<S: 'static>(&self) -> bool {
        self.0
            .as_ref()
            .is_some_and(|state| state.as_ref().as_any().is::<S>())
    }

    pub(super) fn take_state<S: 'static>(&mut self) -> Result<S, Error> {
        Ok(*self
            .0
            .take()
            .ok_or_else(|| {
                Error::ArchitectureModel("prediction target state is already active".into())
            })?
            .into_any()
            .downcast::<S>()
            .expect("prediction target state type checked before transfer"))
    }

    pub(super) fn restore_state<S: MlxStateMechanisms + 'static>(&mut self, state: S) {
        self.0 = Some(Box::new(state));
    }

    pub(crate) fn deep_clone(&self) -> Result<Self, Exception> {
        self.0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?
            .deep_clone_box()
            .map(|state| Self(Some(state)))
    }

    pub(crate) fn restore(&mut self, checkpoint: &Self, stream: &Stream) -> Result<(), Exception> {
        let current = self
            .0
            .as_mut()
            .ok_or_else(|| Exception::custom("prediction target state is already active"))?;
        let checkpoint = checkpoint
            .0
            .as_ref()
            .ok_or_else(|| Exception::custom("prediction target checkpoint is active"))?;
        current.restore_box(checkpoint.as_ref(), stream)
    }

    pub(crate) fn generation(&self) -> Result<u64, Error> {
        let state = self.0.as_ref().ok_or_else(|| {
            Error::ArchitectureModel("prediction target state is already active".into())
        })?;
        u64::try_from(state.offset())
            .map_err(|_| Error::ArchitectureModel("target capture generation is negative".into()))
    }
}

pub(super) struct ExactPredictionCaptureObserver {
    paths: Vec<String>,
    pub(super) values: std::rc::Rc<std::cell::RefCell<Vec<Option<MlxTensor>>>>,
}

impl ExactPredictionCaptureObserver {
    pub(super) fn new(paths: Vec<String>) -> Result<Self, Error> {
        if paths.is_empty() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture declares no target paths".into(),
            ));
        }
        let unique = paths.iter().collect::<std::collections::BTreeSet<_>>();
        if unique.len() != paths.len() {
            return Err(Error::ArchitectureModel(
                "external-assistant capture paths are not unique".into(),
            ));
        }
        let values = std::rc::Rc::new(std::cell::RefCell::new(vec![None; paths.len()]));
        Ok(Self { paths, values })
    }
}

impl eredu_runtime::ActivationObserver<MlxTensor, eredu_nn::Error>
    for ExactPredictionCaptureObserver
{
    fn observe_generated(
        &mut self,
        path: &str,
        _: &MlxTensor,
        _: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<MlxTensor, eredu_nn::Error>,
    ) -> Result<(), eredu_nn::Error> {
        if self.paths.iter().any(|expected| expected == path) {
            self.observe(path, &generate()?)?;
        }
        Ok(())
    }

    fn observe(&mut self, path: &str, value: &MlxTensor) -> Result<(), eredu_nn::Error> {
        if let Some(index) = self.paths.iter().position(|expected| expected == path) {
            if self.values.borrow_mut()[index]
                .replace(value.clone())
                .is_some()
            {
                return Err(eredu_nn::Error::backend(format!(
                    "external-assistant target reached capture path {path} more than once"
                )));
            }
        }
        Ok(())
    }
}

pub(super) struct CompositePredictionTargetOperation<'a> {
    pub(super) operation: ExternalPredictionTargetOperation<'a, MlxTensor>,
}

impl<A>
    eredu_runtime::PredictionTargetOperation<
        PreparedCompositeArchitecture<A>,
        MlxNeuralBackend,
        MlxHybridState,
    > for CompositePredictionTargetOperation<'_>
where
    A: CompositeArchitecture<MlxNeuralBackend, MlxHybridState, Error = eredu_nn::Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = MlxTensor;

    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _state: &mut MlxHybridState,
        parallel: Option<&<MlxNeuralBackend as NeuralBackend>::ParallelContext>,
        context: &Stream,
    ) -> Result<Self::Output, eredu_nn::Error> {
        if parallel.is_some() {
            return Err(eredu_nn::Error::backend(
                "external assistant target operations are unavailable under tensor parallelism",
            ));
        }
        architecture
            .inner_mut()
            .external_prediction_target_operation(self.operation, context)?
            .ok_or_else(|| {
                eredu_nn::Error::backend(
                    "architecture does not implement the selected external target operation",
                )
            })
    }
}

pub(crate) struct MlxEmbeddedPredictionMaterializer;

pub(crate) type MaterializedEmbeddedPrediction =
    eredu_architectures::prediction_extension::MaterializedPredictionExtension<
        MlxNeuralBackend,
        MlxEmbeddedPredictionMaterializer,
    >;

pub(super) fn materialize_prepared_prediction_unit<M>(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
    layout: Option<&eredu_runtime::LocalModelLayout>,
    store: SharedCheckpointSource,
    stream: &Stream,
    _weights_stream: &Stream,
) -> Result<MlxPredictionModule<M>, Error>
where
    M: Parameterized<MlxTensor>,
{
    use crate::backend::runtime::checkpoint::binding::build_mlx_exact_replicated_text_bindings;

    let residency = prepared.residency();
    let shared =
        prepared.role() == eredu_architectures::prediction_extension::PredictionModuleRole::Shared;
    let source_layout = prepared.source_layout().cloned();
    let (source, local, selected_tasks) = prepared.into_parts();
    let mut store = store;
    let mut materialization = eredu_runtime::WeightMaterializationReport::default();
    for group in eredu_runtime::group_replicated_text_transform_tasks(&selected_tasks)
        .map_err(|error| Error::Quantization(error.to_string()))?
    {
        let tasks = group
            .tasks(&selected_tasks)
            .map_err(|error| Error::Quantization(error.to_string()))?;
        let (transformed, report) = quantize_exact_replicated_text_tasks(
            store,
            &source,
            &local,
            &[] as &[M],
            &[],
            source_layout.as_deref(),
            group.quantization(),
            &tasks,
            stream,
        )?;
        store = transformed;
        materialization.merge(report);
    }
    let task_refs = selected_tasks.iter().collect::<Vec<_>>();
    let bindings = build_mlx_exact_replicated_text_bindings(
        &local,
        store.as_ref(),
        &task_refs,
        &std::collections::BTreeSet::new(),
        layout,
    )?;
    let mut parameters = Vec::new();
    let mut declarations = Vec::new();
    super::session::prepared_parameters::collect_module(
        &local,
        &bindings,
        eredu_runtime::parameter_operations::PreparedParameterLocation::Prediction { module: 0 },
        store.as_ref(),
        &mut parameters,
        &mut declarations,
    )?;
    if parameters.len() != declarations.len() {
        return Err(Error::ArchitectureModel(
            "prediction parameters lack exact prepared binding owners".into(),
        ));
    }
    Ok(MlxPredictionModule {
        placeholders: parameters::placeholders(&local),
        replacements: BTreeMap::new(),
        source: store,
        bindings,
        residency,
        shared,
        manager: Arc::new(std::sync::OnceLock::new()),
        id: None,
        stream: stream.clone(),
        inner: local,
        parameters,
        tasks: selected_tasks,
        materialization,
    })
}

pub(crate) fn materialize_prediction_extension(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionExtension<
        MlxNeuralBackend,
    >,
    store: SharedCheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MaterializedEmbeddedPrediction, Error> {
    let mut context = MlxPredictionMaterializationContext {
        store,
        stream,
        weights_stream,
    };
    prepared.materialize::<MlxEmbeddedPredictionMaterializer>(&mut context)
}

pub(crate) struct MlxPredictionMaterializationContext<'a> {
    store: SharedCheckpointSource,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::prediction_extension::PredictionExtensionMaterializer<MlxNeuralBackend>
    for MlxEmbeddedPredictionMaterializer
{
    type Error = Error;
    type Module<M> = MlxPredictionModule<M>;
    type PoolingState = crate::backend::runtime::cache::state::MlxPoolingAttentionCache;
    type SequentialState = crate::backend::runtime::cache::kv::CompressedLatentCache;
    type ModelState = MlxHybridState;
    type Context<'a> = MlxPredictionMaterializationContext<'a>;

    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a MlxTensor>,
        _context: &Stream,
    ) -> Result<(), eredu_core::BackendFailure> {
        let token_validations = active_token_validation_arrays();
        async_eval_with_event(
            values
                .into_iter()
                .map(|value| value.as_array())
                .chain(token_validations.iter()),
        )
        .and_then(|completion| completion.synchronize())
        .and_then(|()| validate_active_token_validations())
        .map_err(eredu_core::BackendFailure::from_error)
    }

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
        layout: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<MlxTensor>,
    {
        materialize_prepared_prediction_unit(
            prepared,
            layout,
            Arc::clone(&context.store),
            context.stream,
            context.weights_stream,
        )
    }

    fn invoke_module<U, O>(
        module: &mut Self::Module<U>,
        context: &Stream,
        operation: impl FnOnce(
            &mut U,
        )
            -> eredu_architectures::prediction_extension::PredictionInvocation<
            MlxTensor,
            O,
        >,
    ) -> Result<O, eredu_nn::Error>
    where
        U: Parameterized<MlxTensor>,
    {
        module
            .invoke(context, |inner| {
                let invocation = operation(inner);
                let roots = invocation.retained_values().cloned().collect();
                (invocation.into_outcome().map_err(Error::from), roots)
            })
            .map_err(|error| match error {
                Error::Neural(error) => error,
                error => eredu_nn::Error::backend_source(error),
            })
    }

    fn pooling_state(
        _context: &mut Self::Context<'_>,
        ordinal: usize,
        policy: eredu_core::cache::LayerCachePolicy,
    ) -> Result<Self::PoolingState, Self::Error> {
        Ok(Self::PoolingState::resident_from_policy(ordinal, &policy)?)
    }

    fn model_state(
        _context: &mut Self::Context<'_>,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self::ModelState, Self::Error> {
        Ok(MlxHybridState::device(layout)?)
    }

    fn sequential_state() -> Self::SequentialState {
        Self::SequentialState::new()
    }

    fn model_memory_observation(
        state: &Self::ModelState,
        additional: u64,
    ) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
        use eredu_runtime::RuntimeStateComponents;
        let retained = state.isolated_snapshot_estimate().map(|e| e.retained_bytes);
        Some(
            eredu_core::speculative::SpeculativePredictionMemoryObservation {
                layer_positions: state
                    .layers()
                    .iter()
                    .map(|layer| u64::try_from(layer.position()).ok())
                    .collect::<Option<Vec<_>>>()?,
                current_state_bytes: retained
                    .and_then(|n| n.checked_add(state.isolated_snapshot_growth(0)?)),
                peak_state_bytes: retained
                    .and_then(|n| n.checked_add(state.isolated_snapshot_growth(additional)?)),
            },
        )
    }

    fn sequential_memory_observation(
        state: &Self::SequentialState,
        _additional: u64,
    ) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
        Some(
            eredu_core::speculative::SpeculativePredictionMemoryObservation {
                layer_positions: vec![u64::try_from(state.offset()).ok()?],
                // Compact snapshot payloads can omit retained backing capacity.
                // A complete native capacity envelope is not yet available here.
                current_state_bytes: None,
                peak_state_bytes: None,
            },
        )
    }

    fn pooling_memory_observation(
        state: &Self::PoolingState,
        _additional: u64,
    ) -> Option<eredu_core::speculative::SpeculativePredictionMemoryObservation> {
        Some(
            eredu_core::speculative::SpeculativePredictionMemoryObservation {
                layer_positions: vec![u64::try_from(state.offset()).ok()?],
                // Compact snapshot payloads can omit retained backing capacity.
                // A complete native capacity envelope is not yet available here.
                current_state_bytes: None,
                peak_state_bytes: None,
            },
        )
    }

    fn pooling_snapshot_estimate(
        state: &Self::PoolingState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut estimate = super::super::speculative::state_snapshot::estimate_state::<
            Self::PoolingState,
        >(state.retained_arrays())?;
        let auxiliary = state
            .residency_manager()
            .map(|manager| manager.isolated_snapshot_bytes())
            .unwrap_or(Some(0))?;
        estimate.retained_bytes = estimate.retained_bytes.checked_add(auxiliary)?;
        estimate.copy_bytes = estimate.copy_bytes.checked_add(auxiliary)?;
        Some(estimate)
    }

    fn pooling_snapshot(
        state: &Self::PoolingState,
        stream: &Stream,
    ) -> Result<Option<Self::PoolingState>, eredu_core::BackendFailure> {
        let copy = (|| {
            super::super::speculative::state_snapshot::settle(state.retained_arrays())?;
            let copy = state.isolated_snapshot(stream)?;
            super::super::speculative::state_snapshot::settle(copy.retained_arrays())?;
            Ok::<_, Exception>(copy)
        })()
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Some(copy))
    }

    fn sequential_snapshot_estimate(
        state: &Self::SequentialState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        if state.is_paged() {
            return None;
        }
        super::super::speculative::state_snapshot::estimate_state::<Self::SequentialState>(
            state.retained_arrays(),
        )
    }

    fn sequential_snapshot(
        state: &Self::SequentialState,
        stream: &Stream,
    ) -> Result<Option<Self::SequentialState>, eredu_core::BackendFailure> {
        if state.is_paged() {
            return Ok(None);
        }
        let copy = (|| {
            super::super::speculative::state_snapshot::settle(state.retained_arrays())?;
            let copy = state.isolated_snapshot(stream)?;
            super::super::speculative::state_snapshot::settle(copy.retained_arrays())?;
            Ok::<_, Exception>(copy)
        })()
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Some(copy))
    }

    fn model_snapshot_estimate(
        state: &Self::ModelState,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        state.isolated_snapshot_estimate()
    }

    fn model_snapshot(
        state: &Self::ModelState,
        stream: &Stream,
    ) -> Result<Option<Self::ModelState>, eredu_core::BackendFailure> {
        if !state.supports_isolated_snapshot() {
            return Ok(None);
        }
        let copy = (|| {
            super::super::speculative::state_snapshot::settle(state.retained_arrays())?;
            let copy = state.isolated_snapshot(stream)?;
            super::super::speculative::state_snapshot::settle(copy.retained_arrays())?;
            Ok::<_, Exception>(copy)
        })()
        .map_err(eredu_core::BackendFailure::from_error)?;
        Ok(Some(copy))
    }
}

impl eredu_architectures::prediction_extension::PredictionModelState<MlxNeuralBackend>
    for MlxHybridState
{
    type LayerState = crate::backend::runtime::cache::state::MlxHybridLayerState;

    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.layers_mut()
    }
}

pub(super) struct SelectedPrediction<P> {
    pub(super) extension: P,
    pub(super) selected: eredu_runtime::SelectedSpeculativeRealization,
}

pub(super) struct NoSelectedPrediction;

pub(super) trait ReplicatedPredictionCapability<A, S, D>: Sized
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
    fn lend(
        model: &mut CompletedReplicatedText<A, S, D, Self>,
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

impl<A, S, D> ReplicatedPredictionCapability<A, S, D> for NoSelectedPrediction
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
    fn lend(
        _: &mut CompletedReplicatedText<A, S, D, Self>,
        _: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }

    fn present() -> bool {
        false
    }
}

pub(crate) trait ErasedExternalPredictionExecutable: 'static {
    fn prepare_external_prediction_target_cache(
        &mut self,
    ) -> Result<MlxPredictionTargetState, Error>;
    fn prefill_external_prediction_target(
        &mut self,
        input: input::ModelInput<'_>,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn verify_external_prediction_target(
        &mut self,
        tokens: &MlxTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut MlxPredictionTargetState,
    ) -> Result<(MlxTensor, ExternalPredictionTargetCapture<MlxTensor>), Error>;
    fn apply_external_prediction_target_operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, MlxTensor>,
    ) -> Result<MlxTensor, Error>;
}

/// Backend-private erased operations for a paired architecture and mutable state.
pub(crate) trait ErasedReplicatedTextExecutable {
    fn projects_final_prefill_position(&self) -> bool {
        false
    }
    fn forecast_state_offset(&self) -> Result<Option<i32>, Exception> {
        Ok(None)
    }
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn prepared_input_plans(
        &self,
        input: input::ModelInput<'_>,
    ) -> Result<
        Vec<eredu_architectures::media_plan::PreparedInputPartPlan>,
        eredu_core::CapabilityError,
    > {
        input
            .parts
            .iter()
            .map(|part| {
                eredu_architectures::media_plan::text_only_input_part(
                    self.effective_model_type(),
                    part,
                    &input::MlxInputInspector,
                )
            })
            .collect()
    }

    fn partition_observation_hooks(
        &self,
    ) -> Option<eredu_runtime::inspection::ObservationHookSupport> {
        None
    }

    fn partition_parameter_description(
        &self,
    ) -> Option<&std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>> {
        None
    }

    fn visit_loaded_parameters(
        &mut self,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<MlxTensor>,
    ) -> bool {
        false
    }
    fn parameter_materialization_tasks(
        &self,
    ) -> &[eredu_runtime::ReplicatedTextMaterializationTask] {
        &[]
    }
    fn prepared_parameter_slots(
        &self,
    ) -> &[eredu_runtime::parameter_operations::PreparedParameterSlot] {
        &[]
    }
    fn with_parameter_slots(
        &mut self,
        _location: &eredu_runtime::parameter_operations::PreparedParameterLocation,
        _selected: &std::collections::BTreeSet<String>,
        _operation: &mut eredu_runtime::parameter_operations::ParameterSlotOperation<
            '_,
            MlxTensor,
            Error,
        >,
        _stream: &Stream,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn publish_parameter_replacements(
        &mut self,
        _values: &std::collections::BTreeMap<String, MlxTensor>,
        _active: bool,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    fn invalidate_parameter_snapshots(&mut self) {}
    fn estimate_parameter_reset_state(
        &self,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        None
    }
    fn prepare_parameter_reset_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "parameter state reset is unavailable".into(),
        ))
    }
    fn exchange_parameter_reset_state(
        &mut self,
        _slot: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "parameter state exchange is unavailable".into(),
        ))
    }
    fn prepare_autoregressive_cache(&mut self) -> Result<MlxPredictionTargetState, Error> {
        Err(Error::Speculative(
            "ordinary prediction state is unavailable".into(),
        ))
    }
    fn autoregressive_forward(
        &mut self,
        _tokens: &Array,
        _cache: &mut MlxPredictionTargetState,
        _prefill: bool,
        _stream: &Stream,
    ) -> Result<Array, Error> {
        Err(Error::Speculative(
            "ordinary prediction state is unavailable".into(),
        ))
    }

    fn native_control_support(&self) -> eredu_core::execution_control::ControlSupport {
        eredu_core::execution_control::ControlSupport::Unsupported {
            reason: "complete native state copying is unavailable for this executable".into(),
        }
    }
    fn estimate_native_control_state(
        &self,
        _saved: Option<&dyn std::any::Any>,
    ) -> Result<Option<eredu_core::execution_control::SnapshotEstimate>, Error> {
        Ok(None)
    }
    fn capture_native_control_state(&mut self) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn estimate_installed_control_growth(&self, _additional: u64) -> Result<Option<u64>, Error> {
        Ok(None)
    }
    fn estimate_native_control_growth(
        &self,
        _saved: &dyn std::any::Any,
        _additional: u64,
    ) -> Result<Option<u64>, Error> {
        Ok(None)
    }
    fn copy_native_control_state(
        &mut self,
        _saved: &dyn std::any::Any,
    ) -> Result<Box<dyn std::any::Any>, Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn validate_native_control_state(&self, _saved: &dyn std::any::Any) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn exchange_native_control_state(
        &mut self,
        _slot: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        Err(Error::ArchitectureModel(
            "native control state is unsupported".into(),
        ))
    }
    fn effective_model_type(&self) -> &str;
    fn capability_estimate(&self) -> &eredu_architectures::capability::CapabilityEstimate;
    #[cfg(test)]
    fn selected_residency(&self) -> eredu_runtime::LayerWeightResidency;
    #[cfg(test)]
    fn state_snapshot(&self) -> StatePresenceSnapshot;
    #[cfg(test)]
    fn fixed_numeric_state_snapshot(&self) -> Result<FixedNumericStateSnapshot, Exception>;
    #[cfg(test)]
    fn checkpoint_restore_probe(
        &mut self,
        tokens: &Array,
        stream: &Stream,
    ) -> Result<CheckpointRestoreProbe, Error>;
    fn residency_report(&self) -> Result<Option<ResidencyReport>, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport>;
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    >;
    fn has_partition_control(&self) -> bool {
        false
    }
    fn partition_sampling_context(
        &self,
    ) -> Option<(
        &crate::backend::runtime::distributed::Group,
        &eredu_runtime::PartitionCommunicationAuthority,
        &Stream,
        usize,
    )> {
        None
    }
    fn partition_public_output(&self) -> bool {
        true
    }
    fn with_embedded_prediction(
        &mut self,
        _continuation: &mut dyn super::prepared_speculative::MlxEmbeddedExecutorContinuation,
    ) -> Option<Result<eredu_core::SpeculativeGenerationBatchOutput, Error>> {
        None
    }
    fn has_embedded_prediction(&self) -> bool {
        false
    }
    fn speculative_activation_execution(
        &self,
    ) -> Option<eredu_architectures::speculative_execution::SpeculativeActivationExecution> {
        None
    }
    fn install_embedded_prediction_observers(
        &mut self,
        _observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        false
    }
    fn take_speculative_activation_capture(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeActivationCapture> {
        None
    }
    fn take_speculative_activation_error(
        &mut self,
    ) -> Option<eredu_core::speculative::SpeculativeControlError> {
        None
    }
    fn external_prediction_mut(
        &mut self,
    ) -> Option<&mut (dyn ErasedExternalPredictionExecutable + 'static)> {
        None
    }
    fn prompt_cache_model_identity(&self) -> &PromptCacheModelIdentity;
    fn reset_cache(&mut self) -> Result<(), Exception>;
    fn reset_cache_distributed(&mut self) -> Result<(), Error>;
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<PromptCacheManifest, Error> {
        let _ = input_identity;
        self.load_prompt_cache(directory, expected, prefix_token_ids)
    }
    fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, Error>;
    fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Error>;
    fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception>;
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error> {
        self.prefill_result_with_observer(
            Ok(input),
            None,
            None,
            stream,
            &mut eredu_runtime::NoopObserver,
        )
    }
    #[cfg(test)]
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        self.decode_result_with_observer(Ok(tokens), stream, &mut eredu_runtime::NoopObserver)
    }
    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error>;
    #[cfg(test)]
    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error> {
        self.prefill_result_with_observer(Ok(input), mask, None, stream, observer)
    }
    fn prefill_result_with_observer(
        &mut self,
        input: Result<input::ModelInput<'_>, Error>,
        mask: Option<&Array>,
        capture_geometry: Option<eredu_core::capture::CaptureRequestShape>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error>;
    fn decode_result_with_observer(
        &mut self,
        tokens: Result<&Array, Error>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Error>,
    ) -> Result<Array, Error>;
}

pub(crate) fn prepared_composite_input(
    input: input::ModelInput<'_>,
) -> Result<eredu_runtime::PreparedModelInput<MlxTensor>, Error> {
    use eredu_runtime::{PreparedInputInspector, PreparedInputPart, PreparedInputPayload};

    input::validate(input)?;
    let parts = input
        .parts
        .iter()
        .map(|part| {
            let payload = match part.payload() {
                input::InputPayload::TokenIds(value) => {
                    PreparedInputPayload::TokenIds(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Tensor(value) => {
                    PreparedInputPayload::Tensor(MlxTensor::from_array(value.clone()))
                }
                input::InputPayload::Embeddings(value) => {
                    PreparedInputPayload::Embeddings(MlxTensor::from_array(value.clone()))
                }
                _ => {
                    return Err(eredu_core::PreparedInputError::BackendTensorIdentity(
                        "MLX prepared input contains an unknown payload kind".into(),
                    ))
                }
            };
            PreparedInputPart::new_with_extents(
                part.modality(),
                payload,
                part.metadata()
                    .iter()
                    .map(|(key, value)| (*key, MlxTensor::from_array(value.clone()))),
                part.extents().iter().copied(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let inspector = input::MlxTensorInputInspector;
    eredu_runtime::PreparedModelInput::new(parts, |tensor| inspector.identity(tensor))
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

#[cfg(test)]
mod completion_tests {
    use super::*;
    use crate::backend::{
        nn::tensor::{validate_token_domain, TokenValidationScope},
        ExecutionContext,
    };
    use eredu_architectures::prediction_extension::PredictionExtensionMaterializer;
    use safemlx::{Device, DeviceType};
    use std::error::Error as _;

    fn prove(device: DeviceType) {
        let execution = ExecutionContext::new(Device::new(device, 0));
        let stream = execution.stream();
        for token in [2, 4] {
            let scope = TokenValidationScope::begin().unwrap();
            let tokens = Array::from_slice(&[token], &[1]);
            let _validated = validate_token_domain(&tokens, 4, None, stream)
                .expect("deferred token checks must not fail during graph construction");
            // The assertion is independent of the retained tensor. Completion must
            // settle both without relying on an output's dependency graph.
            let retained = MlxTensor::from_array(
                Array::from_slice(&[2.0_f32, -3.0], &[2])
                    .multiply(Array::from_f32(3.0), stream)
                    .unwrap(),
            );
            let result =
                MlxEmbeddedPredictionMaterializer::complete_prediction_values([&retained], stream);
            assert_eq!(
                retained
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_as_slice::<f32>()
                    .unwrap(),
                &[6.0, -9.0]
            );
            if token == 2 {
                result.unwrap();
            } else {
                let failure = result.unwrap_err();
                let source = failure
                    .source()
                    .unwrap()
                    .downcast_ref::<Exception>()
                    .expect("native exception must remain the error source");
                assert!(source.to_string().contains("token ID is outside 0..4"));
            }
            drop(scope);
        }
    }

    #[test]
    fn prediction_completion_retains_dependencies_and_errors_cpu() {
        prove(DeviceType::Cpu);
    }
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a local MLX Metal device"]
    fn prediction_completion_retains_dependencies_and_errors_metal() {
        prove(DeviceType::Gpu);
    }
}
