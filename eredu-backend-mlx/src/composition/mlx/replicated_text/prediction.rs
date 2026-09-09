use super::*;

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
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    >;
}

impl MlxParameterBankTelemetry for eredu_runtime::DirectReplicatedTextExecution {
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<
        eredu_architectures::PlannedResidentGatedProduct,
    >
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

impl MlxParameterBankTelemetry
    for eredu_runtime::RoutedReplicatedTextExecution<eredu_architectures::PlannedResidentRelu2>
{
    fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
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
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        Ok(None)
    }
}

pub(super) type MlxAddressableGated = eredu_architectures::PlannedAddressableGatedProduct<
    MlxNeuralBackend,
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank,
    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
>;
pub(super) type MlxAddressableRelu2 = eredu_architectures::PlannedAddressableRelu2<
    MlxNeuralBackend,
    crate::backend::runtime::residency::parameter_bank::AddressableParameterBank,
    crate::backend::runtime::residency::parameter_bank::MlxIndexedMovement,
>;

macro_rules! addressable_bank_telemetry {
    ($provider:ty) => {
        impl MlxParameterBankTelemetry
            for eredu_runtime::RoutedReplicatedTextExecution<$provider>
        {
            fn parameter_bank_report(
                &self,
            ) -> Result<
                Option<
                    crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport,
                >,
                Error,
            > {
                self.provider()
                    .bank_report()
                    .map(Some)
                    .map_err(|error| Error::ArchitectureModel(error.to_string()))
            }
        }
    };
}

addressable_bank_telemetry!(MlxAddressableGated);
addressable_bank_telemetry!(MlxAddressableRelu2);

pub(super) trait ErasedPredictionTargetState: std::any::Any {
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
    store: &dyn CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<crate::backend::nn::shared::MlxModule<M>, Error>
where
    M: Parameterized<MlxTensor>,
{
    use crate::backend::runtime::checkpoint::binding::{
        build_mlx_exact_replicated_text_bindings, materialize_module_bindings,
        populate_module_from_arrays_excluding,
    };

    let (source, mut local, selected_tasks) = prepared.into_parts();
    let task_refs = selected_tasks.iter().collect::<Vec<_>>();
    let bindings = build_mlx_exact_replicated_text_bindings(
        &source,
        store,
        &task_refs,
        &std::collections::BTreeSet::new(),
        None,
    )?;
    let bindings = match layout {
        Some(layout) => shard_unmaterialized_bindings(
            bindings,
            store,
            layout,
            &std::collections::BTreeSet::new(),
        )?,
        None => bindings,
    };
    let arrays = materialize_module_bindings(store, &bindings, weights_stream, stream)?;
    populate_module_from_arrays_excluding(&mut local, &arrays, |_| false)?;
    Ok(crate::backend::nn::shared::MlxModule::new(local))
}

pub(crate) fn materialize_prediction_extension(
    prepared: eredu_architectures::prediction_extension::PreparedPredictionExtension<
        MlxNeuralBackend,
    >,
    store: &dyn CheckpointSource,
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
    store: &'a dyn CheckpointSource,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::prediction_extension::PredictionExtensionMaterializer<MlxNeuralBackend>
    for MlxEmbeddedPredictionMaterializer
{
    type Error = Error;
    type Module<M> = crate::backend::nn::shared::MlxModule<M>;
    type PoolingState = crate::backend::runtime::cache::state::MlxPoolingAttentionCache;
    type SequentialState = crate::backend::runtime::cache::kv::CompressedLatentCache;
    type ModelState = MlxHybridState;
    type Context<'a> = MlxPredictionMaterializationContext<'a>;

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
            context.store,
            context.stream,
            context.weights_stream,
        )
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
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
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
    fn install_embedded_prediction_observers(
        &mut self,
        _observers: MlxEmbeddedPredictionObservers,
    ) -> bool {
        false
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
    fn prefill(&mut self, input: input::ModelInput<'_>, stream: &Stream) -> Result<Array, Error>;
    fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error>;
    #[cfg(test)]
    fn forward_with_observer(
        &mut self,
        tokens: &Array,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
    fn prefill_with_observer(
        &mut self,
        input: input::ModelInput<'_>,
        mask: Option<&Array>,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
    ) -> Result<Array, Error>;
    fn decode_with_observer(
        &mut self,
        tokens: &Array,
        stream: &Stream,
        observer: &mut dyn eredu_runtime::ActivationObserver<Array, Exception>,
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
