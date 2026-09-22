use eredu_architectures::composite_execution::{
    CompositeArchitecture, ExternalPredictionCaptureRequest, ExternalPredictionTargetCapture,
    ExternalPredictionTargetOperation, PreparedCompositeArchitecture, PreparedCompositeInput,
};
use eredu_core::ModelConfigurationResolver as _;
use eredu_runtime::{
    DeviceState, PredictionTargetOperation, ReplicatedTextMaterializationTask,
    ReplicatedTextSession, ReplicatedTextSessionMechanisms, ResidentUnitWindow,
};

type ReferenceState = DeviceState<ReferenceBackend, ReferenceCache>;

#[derive(Clone)]
struct ReferencePredictionState {
    layout: eredu_runtime::StateLayout,
    layers: Vec<ReferenceCache>,
}

impl eredu_runtime::RuntimeState<ReferenceBackend> for ReferencePredictionState {
    type RetainedValues<'a> = std::iter::Empty<&'a ReferenceTensor>;

    fn layout(&self) -> &eredu_runtime::StateLayout {
        &self.layout
    }

    fn visit_all_retained_values(
        &self,
        visitor: &mut dyn FnMut(&ReferenceTensor),
    ) -> Result<(), eredu_runtime::StateError> {
        for layer in &self.layers {
            for value in
                eredu_runtime::RuntimeLayerState::<ReferenceBackend>::retained_values(layer)
            {
                visitor(value);
            }
        }
        Ok(())
    }

    fn retained_values(
        &self,
        _ordinal: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, eredu_runtime::StateError> {
        Ok(std::iter::empty())
    }
}

impl eredu_architectures::prediction_extension::PredictionModelState<ReferenceBackend>
    for ReferencePredictionState
{
    type LayerState = ReferenceCache;

    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        &mut self.layers
    }
}

struct ReferencePredictionModule<M>(M);

impl<M> AsMut<M> for ReferencePredictionModule<M> {
    fn as_mut(&mut self) -> &mut M {
        &mut self.0
    }
}

struct ReferencePredictionMaterializer;

/// Snapshot authority deliberately has a different type from the tensor stream.
#[derive(Clone, Copy, Default)]
struct ReferencePredictionSnapshotContext<'a> {
    marker: u64,
    observed: Option<&'a RefCell<Vec<(&'static str, u64)>>>,
}

impl ReferencePredictionSnapshotContext<'_> {
    fn record(self, member: &'static str) {
        if let Some(observed) = self.observed {
            observed.borrow_mut().push((member, self.marker));
        }
    }
}

#[derive(Clone, Copy)]
struct ReferenceEmbeddedContext<'a> {
    stream: &'a (),
    snapshot: ReferencePredictionSnapshotContext<'a>,
    schedule: Option<&'a eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority>,
}

impl Default for ReferenceEmbeddedContext<'_> {
    fn default() -> Self {
        Self {
            stream: &(),
            snapshot: ReferencePredictionSnapshotContext::default(),
            schedule: None,
        }
    }
}

struct ReferencePredictionMaterializationContext<'a> {
    store: &'a dyn eredu_checkpoint::store::CheckpointSource,
}

impl eredu_architectures::prediction_extension::PredictionExtensionMaterializer<ReferenceBackend>
    for ReferencePredictionMaterializer
{
    type Error = Error;
    type Module<M> = ReferencePredictionModule<M>;
    type PoolingState = ReferenceCache;
    type SequentialState = ReferenceCache;
    type ModelState = ReferencePredictionState;
    type Context<'a> = ReferencePredictionMaterializationContext<'a>;
    type SnapshotContext<'a> = ReferencePredictionSnapshotContext<'a>;

    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a ReferenceTensor>,
        _context: &<ReferenceTensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        // The reference tensors are eager; consume the complete dependency set.
        for _value in values {}
        Ok(())
    }

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
        _: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<ReferenceTensor>,
    {
        let (_, mut local, tasks) = prepared.into_parts();
        let mut recipes = tasks
            .iter()
            .map(|task| {
                task.source_recipe()
                    .map(|recipe| (task.name().to_owned(), recipe))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>, _>>()
            .map_err(|error| Error::backend(error.to_string()))?;
        for companion in tasks.iter().flat_map(|task| task.output_companions()) {
            if let Some(task) = companion.materialization_task() {
                recipes.insert(
                    companion.name().to_owned(),
                    task.source_recipe()
                        .map_err(|error| Error::backend(error.to_string()))?,
                );
            } else if let Some(recipe) = companion.derived_recipe() {
                recipes.insert(companion.name().to_owned(), recipe.clone());
            }
        }
        struct Bindings<'a> {
            values: Vec<eredu_runtime::WeightBinding>,
            missing: Vec<String>,
            recipes: &'a mut std::collections::BTreeMap<
                String,
                eredu_checkpoint::recipe::DerivedWeightRecipe,
            >,
        }
        impl<'a, 'r> ParameterVisitor<'a, ReferenceTensor> for Bindings<'r> {
            fn visit(
                &mut self,
                metadata: eredu_nn::ParameterMetadataView<'_>,
                value: &'a ReferenceTensor,
            ) {
                let bytes = value
                    .shape()
                    .iter()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .product::<u64>()
                    * 4;
                let binding = self.recipes.remove(metadata.id().as_str()).map(|recipe| {
                    eredu_runtime::WeightBinding::from_recipe(metadata.id().as_str(), recipe, bytes)
                });
                if let Some(binding) = binding {
                    self.values.push(binding.unwrap());
                } else {
                    self.missing.push(metadata.id().as_str().to_owned());
                }
            }
        }
        let mut bindings = Bindings {
            values: Vec::new(),
            missing: Vec::new(),
            recipes: &mut recipes,
        };
        local.visit_parameters(&mut bindings).map_err(Error::backend)?;
        let values = std::mem::take(&mut bindings.values);
        let missing = std::mem::take(&mut bindings.missing);
        drop(bindings);
        if !recipes.is_empty() || !missing.is_empty() || values.is_empty() {
            return Err(Error::backend(format!(
                "reference prediction exact tasks disagree with module topology: remaining={:?}, missing={missing:?}",
                recipes.keys().collect::<Vec<_>>(),
            )));
        }
        let materialized =
            eredu_runtime::materialize_bindings::<ReferenceBackend>(context.store, &values, &())
                .map_err(|error| Error::backend(error.to_string()))?;
        eredu_runtime::bind_materialized_unit::<ReferenceBackend, _>(&mut local, materialized)
            .map_err(|error| Error::backend(error.to_string()))?;
        Ok(ReferencePredictionModule(local))
    }

    fn pooling_state(
        _: &mut Self::Context<'_>,
        _: usize,
        policy: eredu_core::cache::LayerCachePolicy,
    ) -> Result<Self::PoolingState, Self::Error> {
        Ok(ReferenceCache {
            offset: 0,
            window: policy
                .attention()
                .and_then(|attention| attention.window())
                .map(|v| v.get() as i32),
            resets: 0,
            fixed: None,
        })
    }

    fn model_state(
        _: &mut Self::Context<'_>,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self::ModelState, Self::Error> {
        let layers = layout
            .layers()
            .iter()
            .map(|policy| ReferenceCache {
                offset: 0,
                window: policy
                    .attention()
                    .and_then(|attention| attention.window())
                    .map(|v| v.get() as i32),
                resets: 0,
                fixed: None,
            })
            .collect();
        Ok(ReferencePredictionState { layout, layers })
    }

    fn sequential_state() -> Self::SequentialState {
        ReferenceCache {
            offset: 0,
            window: None,
            resets: 0,
            fixed: None,
        }
    }

    fn sequential_snapshot<'a>(
        state: &Self::SequentialState,
        context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::SequentialState>, eredu_core::BackendFailure> {
        context.record("sequential");
        Ok(Some(state.clone()))
    }

    fn pooling_snapshot<'a>(
        state: &Self::PoolingState,
        context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::PoolingState>, eredu_core::BackendFailure> {
        context.record("pooling");
        Ok(Some(state.clone()))
    }

    fn model_snapshot<'a>(
        state: &Self::ModelState,
        context: Self::SnapshotContext<'a>,
    ) -> Result<Option<Self::ModelState>, eredu_core::BackendFailure> {
        context.record("model");
        Ok(Some(state.clone()))
    }
}

struct ReferenceReplicatedMechanisms;

#[allow(dead_code)]
enum ReferencePrefillGuard {
    Inference(eredu_runtime::working_memory::InferenceRequest),
    Speculative(eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority),
}

impl<A> ReplicatedTextSessionMechanisms<A, ReferenceBackend> for ReferenceReplicatedMechanisms
where
    A: eredu_runtime::LayeredArchitecture<ReferenceBackend, ReferenceState, Error = Error>,
{
    fn prefill_state_frontier(&self, state: &Self::State) -> Result<Option<u64>, Self::Error> {
        state
            .as_ref()
            .first()
            .map(|layer| u64::try_from(layer.offset).map_err(Error::backend))
            .transpose()
    }

    type PrefillReservationGuard = ReferencePrefillGuard;
    fn begin_prefill_reservation(
        &mut self,
        reservation: eredu_runtime::working_memory::InferenceRequest,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        Ok(ReferencePrefillGuard::Inference(reservation))
    }
    fn coordinate_speculative_prefill_entry(
        &mut self,
        authority: &eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority,
        role: Option<eredu_runtime::prefill::PrefillControlRole>,
    ) -> Result<Option<Self::PrefillReservationGuard>, Self::Error> {
        if matches!(
            role,
            None | Some(eredu_runtime::prefill::PrefillControlRole::FinalIndex)
        ) {
            return Ok(None);
        }
        let controls = std::mem::size_of::<Self::PrefillReservationGuard>()
            + std::mem::size_of::<Option<Self::PrefillReservationGuard>>()
            + std::mem::size_of::<Result<Option<Self::PrefillReservationGuard>, Error>>();
        authority
            .metadata_funding()
            .reserve_metadata(controls)
            .map_err(Error::backend_retained_source)?;
        Ok(Some(ReferencePrefillGuard::Speculative(authority.clone())))
    }
    fn finish_prefill_reservation(
        &mut self,
        _guard: Self::PrefillReservationGuard,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    type PromptCacheManifest = eredu_core::cache::PromptCacheManifest;
    type State = ReferenceState;
    type PolicyError = eredu_runtime::ResidentUnitWindowError;
    type ResidentPolicy = ResidentUnitWindow<A::Unit>;
    type BoundedPolicy = ResidentUnitWindow<A::Unit>;
    type StateCheckpoint = ReferenceState;
    type StateReport = ();
    type ExecutionReport = ();
    type Error = Error;

    fn prepare_materialization(
        &mut self,
        _: &mut A,
        _: &eredu_runtime::ExecutionUnitLayout,
        _: &mut [A::Unit],
        _: Option<&mut A>,
        _: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        _: &[String],
        _: &(),
    ) -> Result<(), Self::Error> {
        if tasks.is_empty() {
            return Err(Error::backend(
                "reference construction received no materialization tasks",
            ));
        }
        Ok(())
    }

    fn realize_state(
        &mut self,
        selected: &eredu_runtime::SelectedStateRealization,
        _: &(),
    ) -> Result<Self::State, Self::Error> {
        DeviceState::create(selected.layout().clone(), |_, policy| {
            Ok::<_, Error>(ReferenceCache {
                offset: 0,
                window: policy
                    .attention()
                    .and_then(|attention| attention.window())
                    .map(|window| window.get() as i32),
                resets: 0,
                fixed: None,
            })
        })
    }

    fn resident_policy(
        &mut self,
        _: &mut A,
        units: Vec<A::Unit>,
        _: &eredu_runtime::SelectedReplicatedTextRealization,
        _: &(),
    ) -> Result<Self::ResidentPolicy, Self::Error> {
        Ok(ResidentUnitWindow::new(units))
    }

    fn bounded_policy(
        &mut self,
        _: &mut A,
        _: &eredu_runtime::SelectedReplicatedTextRealization,
        _: &(),
    ) -> Result<Self::BoundedPolicy, Self::Error> {
        Ok(ResidentUnitWindow::new(Vec::new()))
    }

    fn index_text_output(
        &mut self,
        mut output: ReferenceTensor,
        sequence_index: i32,
        _: &(),
    ) -> Result<ReferenceTensor, Self::Error> {
        if sequence_index != -1 || output.0.len() != 3 || output.0[1] <= 0 {
            return Err(Error::backend("invalid reference text output selection"));
        }
        output.0[1] = 1;
        Ok(output)
    }

    fn copy_checkpoint_state(
        &mut self,
        state: &Self::State,
        _: &(),
    ) -> Result<Self::StateCheckpoint, Self::Error> {
        Ok(state.clone())
    }

    fn restore_checkpoint_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        _: &(),
    ) -> Result<(), Self::Error> {
        *state = checkpoint;
        Ok(())
    }

    fn load_prompt_cache(
        &mut self,
        _: &Self::State,
        _: &std::path::Path,
        _: &eredu_core::cache::PromptCacheDescriptor,
        _: &eredu_core::cache::PromptCacheModelIdentity,
        _: &[u32],
        _: &eredu_runtime::SelectedStateRealization,
        _: &(),
    ) -> Result<(Self::State, eredu_core::cache::PromptCacheManifest), Self::Error> {
        Err(Error::backend(
            "reference prompt-cache loading is unavailable",
        ))
    }

    fn save_prompt_cache(
        &mut self,
        _: &mut Self::State,
        _: &std::path::Path,
        _: eredu_core::cache::PromptCacheDescriptor,
        _: &[u32],
        _: &eredu_core::cache::PromptCacheOptions,
        _: &(),
    ) -> Result<eredu_core::cache::PromptCacheManifest, Self::Error> {
        Err(Error::backend(
            "reference prompt-cache saving is unavailable",
        ))
    }

    fn state_report(&self, _: &Self::State) -> Result<Self::StateReport, Self::Error> {
        Ok(())
    }

    fn execution_report(
        &self,
        _: eredu_runtime::LayerWeightResidency,
        _: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Self::Error> {
        Ok(())
    }

    fn complete(
        &mut self,
        _: Option<&ReferenceTensor>,
        _: &Self::State,
        _: &(),
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct ReferenceInputInspector;

impl eredu_runtime::PreparedInputInspector<ReferenceTensor> for ReferenceInputInspector {
    fn identity(
        &self,
        tensor: &ReferenceTensor,
    ) -> Result<eredu_core::InputTensorIdentity, eredu_core::PreparedInputError> {
        eredu_core::InputTensorIdentity::new(
            eredu_core::checkpoint::TensorDtype::U32,
            tensor
                .shape()
                .iter()
                .map(|dimension| usize::try_from(*dimension).unwrap())
                .collect(),
        )
    }

    fn i32_values(&self, _: &ReferenceTensor) -> Result<Vec<i32>, eredu_core::CapabilityError> {
        Err(eredu_core::CapabilityError::Observation(
            "reference token input has no metadata values".into(),
        ))
    }

    fn bool_values(&self, _: &ReferenceTensor) -> Result<Vec<bool>, eredu_core::CapabilityError> {
        Err(eredu_core::CapabilityError::Observation(
            "reference token input has no metadata values".into(),
        ))
    }
}

#[derive(Clone)]
struct ReferencePreparedInput {
    input: eredu_runtime::PreparedModelInput<ReferenceTensor>,
    identity: eredu_runtime::PreparedInputCacheIdentity,
    chunk: Option<std::num::NonZeroU64>,
    external_schedule: Option<(
        eredu_runtime::SelectedSpeculativeRealization,
        eredu_runtime::speculative::external_occurrence::ExternalPredictionShape,
        SpeculativeConfig,
    )>,
}

impl ReferencePreparedInput {
    fn tokens(tokens: &[u32]) -> Result<Self, Error> {
        let width = i32::try_from(tokens.len()).map_err(Error::backend)?;
        let part = eredu_runtime::PreparedInputPart::new(
            eredu_core::InputModality::Text,
            eredu_runtime::PreparedInputPayload::TokenIds(ReferenceTensor(vec![1, width])),
            [],
        )
        .map_err(|error| Error::backend(error.to_string()))?;
        let input = eredu_runtime::PreparedModelInput::new(vec![part], |tensor| {
            eredu_runtime::PreparedInputInspector::identity(&ReferenceInputInspector, tensor)
        })
        .map_err(|error| Error::backend(error.to_string()))?;
        let identity = input
            .cache_identity(format!("reference-tokens-{tokens:?}"))
            .map_err(|error| Error::backend(error.to_string()))?;
        Ok(Self {
            input,
            identity,
            chunk: None,
            external_schedule: None,
        })
    }
}

struct ReferencePredictionInput;

impl<A, S>
    eredu_architectures::speculative_execution::ReplicatedPredictionInput<
        A,
        ReferenceBackend,
        S,
        Error,
    > for ReferencePredictionInput
where
    S: eredu_runtime::RuntimeState<ReferenceBackend>,
    A: eredu_runtime::ReplicatedTextArchitecture<ReferenceBackend, S, Error = Error>,
{
    type Input = ReferencePreparedInput;

    type Prefill =
        eredu_architectures::speculative_execution::TextPredictionPrefill<ReferenceTensor>;

    fn with_prefill_source<R>(
        &mut self,
        input: Self::Input,
        _: &(),
        operation: impl FnOnce(Result<Self::Prefill, Error>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let prepared = input
            .input
            .parts()
            .iter()
            .map(|part| match part.payload() {
                eredu_runtime::PreparedInputPayload::TokenIds(tokens) => Ok(tokens.clone()),
                _ => Err(Error::backend(
                    "reference prediction input is not token IDs",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|tokens| {
                eredu_architectures::speculative_execution::TextPredictionPrefill::new(
                    tokens,
                    Some(eredu_runtime::SharedPreparedInputCacheIdentity::new(
                        input.identity.clone(),
                    )),
                    input.chunk,
                )
            });
        operation(prepared)
    }

    fn requested_chunks(input: &Self::Input) -> Option<std::num::NonZeroU64> {
        input.chunk
    }

    fn with_prefill<R>(
        &mut self,
        input: Self::Input,
        _: &(),
        operation: impl for<'a> FnOnce(
            A::Input<'a>,
            ReferenceTensor,
            Option<&'a eredu_runtime::PreparedInputCacheIdentity>,
        ) -> Result<R, Error>,
    ) -> Result<R, Error> {
        let tokens = input
            .input
            .parts()
            .first()
            .and_then(|part| match part.payload() {
                eredu_runtime::PreparedInputPayload::TokenIds(tokens) => Some(tokens.clone()),
                _ => None,
            })
            .ok_or_else(|| Error::backend("reference prediction input is not text tokens"))?;
        operation(
            A::text_input(&tokens, None),
            tokens.clone(),
            Some(&input.identity),
        )
    }

    fn with_decode<R>(
        &mut self,
        tokens: &ReferenceTensor,
        _: &(),
        operation: impl for<'a> FnOnce(A::Input<'a>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        operation(A::text_input(tokens, None))
    }
}

struct ReferenceEmbeddedMechanisms;
struct ReferenceEmbeddedExecutorTypes;

impl eredu_architectures::speculative_execution::EmbeddedExecutorTypes
    for ReferenceEmbeddedExecutorTypes
{
    type Input = ReferencePreparedInput;
    type Logits = u32;
    type Context<'a> = ReferenceEmbeddedContext<'a>;
    type Completion = ReferenceExternalCompletion;
    type Telemetry = ();
    type Error = Error;

    fn erased_type_mismatch(value: &'static str) -> Self::Error {
        Error::backend(format!("reference embedded executor mismatched {value}"))
    }
}

impl<A, S>
    eredu_architectures::speculative_execution::ReplicatedPredictionNative<
        A,
        ReferenceBackend,
        S,
        ReferenceEmbeddedMechanisms,
    > for ReferencePredictionMaterializer
where
    S: eredu_runtime::LayerRuntimeState<ReferenceBackend> + Clone,
    S::LayerState: eredu_runtime::RuntimeStateComponents<ReferenceBackend>,
    A: eredu_runtime::LayeredArchitecture<ReferenceBackend, S, Error = Error>,
{
    type Input = ReferencePreparedInput;
    type Telemetry = ();
    type ExecutorTypes = ReferenceEmbeddedExecutorTypes;

    fn prefill_schedule_authority(
        context: ReferenceEmbeddedContext<'_>,
    ) -> Result<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority, Error> {
        context.schedule.cloned().ok_or_else(|| {
            Error::backend_retained_source(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )
        })
    }

    fn executor_context<'a>(
        context: <Self::ExecutorTypes as eredu_architectures::speculative_execution::EmbeddedExecutorTypes>::Context<'a>,
    ) -> <ReferenceEmbeddedMechanisms as eredu_architectures::speculative_execution::SpeculativeTensorMechanisms>::Context<'a>{
        context
    }

    fn target_context<'a>(
        context: <ReferenceEmbeddedMechanisms as eredu_architectures::speculative_execution::SpeculativeTensorMechanisms>::Context<'a>,
    ) -> &'a () {
        context.stream
    }

    fn prediction_snapshot_context<'a>(
        context: <ReferenceEmbeddedMechanisms as eredu_architectures::speculative_execution::SpeculativeTensorMechanisms>::Context<'a>,
    ) -> <Self as eredu_architectures::prediction_extension::PredictionExtensionMaterializer<
        ReferenceBackend,
    >>::SnapshotContext<'a>
    where
        Self: eredu_architectures::prediction_extension::PredictionExtensionMaterializer<
            ReferenceBackend,
        >,
        ReferenceBackend: eredu_nn::BlockwiseAttentionBackend
            + eredu_nn::DistributedNeuralBackend
            + eredu_nn::GroupedNeuralBackend
            + eredu_nn::HyperNeuralBackend,
    {
        context.snapshot
    }

    fn control_state_snapshot<'a>(
        state: &S,
        context: ReferenceEmbeddedContext<'a>,
    ) -> Result<Option<S>, eredu_core::speculative::SpeculativeControlError> {
        context.snapshot.record("target");
        Ok(Some(state.clone()))
    }

    fn checkpoint(state: &S) -> Result<S, Error> {
        Ok(state.clone())
    }

    fn restore(state: &mut S, checkpoint: &S, _: &()) -> Result<(), Error> {
        state.clone_from(checkpoint);
        Ok(())
    }

    fn generation(state: &S) -> Result<u64, Error> {
        let mut checkpoint = state.clone();
        let position = checkpoint
            .layer(0)
            .map_err(|error| Error::backend(error.to_string()))?
            .position();
        u64::try_from(position).map_err(Error::backend)
    }

    fn token(_: u32, _: ReferenceEmbeddedContext<'_>) -> Result<ReferenceTensor, Error> {
        Ok(ReferenceTensor(vec![1, 1]))
    }

    fn shape(tensor: &ReferenceTensor) -> &[i32] {
        tensor.shape()
    }

    fn validate<T>(operation: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        operation()
    }

    fn session_error(error: impl std::fmt::Display) -> Error {
        Error::backend(error.to_string())
    }

    fn session_failure(error: eredu_core::BackendFailure) -> Error {
        Error::backend_retained_source(error)
    }

    fn take_telemetry() -> Result<Self::Telemetry, Error> {
        Ok(())
    }
}

impl eredu_architectures::speculative_execution::SpeculativeTensorMechanisms
    for ReferenceEmbeddedMechanisms
{
    type Tensor = ReferenceTensor;
    type Logits = u32;
    type Context<'a> = ReferenceEmbeddedContext<'a>;
    type Completion = ReferenceExternalCompletion;
    type Error = Error;

    fn observation_error(message: &'static str) -> Self::Error {
        Error::backend(message)
    }

    fn empty_prediction_input() -> Self::Error {
        Error::backend("reference embedded input is empty")
    }

    fn fused_prediction_exhausted() -> Self::Error {
        Error::backend("reference fused prediction is exhausted")
    }

    fn invalid_prediction_commit(verified: usize, available: usize) -> Self::Error {
        Error::backend(format!("invalid embedded commit {verified}/{available}"))
    }

    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Self::Error {
        Error::backend(format!(
            "invalid embedded output {logits}/{capture}/{tokens}/{expected:?}"
        ))
    }

    fn invalid_fused_capacity(requested: usize, available: usize) -> Self::Error {
        Error::backend(format!("invalid fused capacity {requested}/{available}"))
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        value
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Error::backend("reference embedded tensor has no sequence axis"))
            .and_then(|value| usize::try_from(value).map_err(Error::backend))
    }

    fn prefill_score_layout<'a>(
        _: Self::Context<'a>,
    ) -> eredu_runtime::replicated_session::PrefillScoreLayout {
        eredu_runtime::replicated_session::PrefillScoreLayout::SelectedPositions
    }

    fn selected_prefill_logits(_: Self::Tensor) -> Result<Self::Logits, Self::Error> {
        Ok(2)
    }

    fn logits_row<'a>(
        _: &Self::Tensor,
        _: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(2)
    }

    fn tensor_row<'a>(
        value: &Self::Tensor,
        row: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        if row >= Self::sequence_len(value)? {
            return Err(Error::backend("reference embedded row is out of bounds"));
        }
        let mut shape = value.shape().to_vec();
        shape[1] = 1;
        Ok(ReferenceTensor(shape))
    }

    fn tensor_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut shape = value.shape().to_vec();
        shape[1] = i32::try_from(end).map_err(Error::backend)?;
        Ok(ReferenceTensor(shape))
    }

    fn token_range<'a>(
        _: &Self::Tensor,
        start: usize,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            1,
            i32::try_from(end.saturating_sub(start)).map_err(Error::backend)?,
        ]))
    }

    fn token_prefix<'a>(
        _: &Self::Tensor,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            1,
            i32::try_from(end).map_err(Error::backend)?,
        ]))
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            1,
            i32::try_from(tokens.len()).map_err(Error::backend)?,
        ]))
    }

    fn fused_logits_row<'a>(
        _: &Self::Tensor,
        _: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Logits, Self::Error> {
        Ok(2)
    }

    fn submit_verification_completion<'a>(
        output: &eredu_architectures::speculative_execution::EmbeddedPredictionOutput<Self::Tensor>,
        inputs: &Self::Tensor,
        _: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        Ok(ReferenceExternalCompletion::submit([
            output.logits().clone(),
            output.capture().clone(),
            output.tokens().clone(),
            inputs.clone(),
        ]))
    }
}

struct ExactReferenceCaptureObserver {
    paths: Vec<String>,
    values: Rc<RefCell<Vec<Option<ReferenceTensor>>>>,
}

impl ExactReferenceCaptureObserver {
    fn new(paths: Vec<String>) -> Self {
        let values = Rc::new(RefCell::new(vec![None; paths.len()]));
        Self { paths, values }
    }
}

impl eredu_runtime::ActivationObserver<ReferenceTensor, Error> for ExactReferenceCaptureObserver {
    fn requires_sequence_readout(&self) -> bool {
        self.paths
            .iter()
            .any(|path| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
    }
    fn observe(&mut self, path: &str, value: &ReferenceTensor) -> Result<(), Error> {
        if let Some(index) = self.paths.iter().position(|expected| expected == path) {
            if self.values.borrow_mut()[index]
                .replace(value.clone())
                .is_some()
            {
                return Err(Error::backend(format!(
                    "reference target reached capture path {path} twice"
                )));
            }
        }
        Ok(())
    }
}

struct ReferenceCompositeOperation<'a> {
    operation: ExternalPredictionTargetOperation<'a, ReferenceTensor>,
}

impl<A>
    PredictionTargetOperation<PreparedCompositeArchitecture<A>, ReferenceBackend, ReferenceState>
    for ReferenceCompositeOperation<'_>
where
    A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
{
    type Output = ReferenceTensor;

    fn preserves_architecture_declarations(&self) -> bool {
        true
    }

    fn apply(
        self,
        architecture: &mut PreparedCompositeArchitecture<A>,
        _: &mut ReferenceState,
        parallel: Option<&<ReferenceBackend as NeuralBackend>::ParallelContext>,
        context: &(),
    ) -> Result<Self::Output, Error> {
        if parallel.is_some() {
            return Err(Error::backend(
                "reference external target operation unexpectedly used tensor parallelism",
            ));
        }
        architecture
            .inner_mut()
            .external_prediction_target_operation(self.operation, context)?
            .ok_or_else(|| Error::backend("external target operation was unavailable"))
    }
}

trait ReferenceExternalTarget {
    fn profile(&self) -> eredu_architectures::external_assistant::ExternalAssistantTargetProfile;
    fn prepare_cache(&mut self) -> Result<ReferenceState, Error>;
    fn prefill(
        &mut self,
        input: ReferencePreparedInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
    ) -> Result<
        (
            ReferenceTensor,
            ExternalPredictionTargetCapture<ReferenceTensor>,
        ),
        Error,
    >;
    fn prefill_spans(
        &mut self,
        input: ReferencePreparedInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
            ReferenceTensor,
            Error,
        >,
        cancellation: &GenerationCancellationToken,
        origin: eredu_core::speculative::SpeculativeActivationOrigin,
    ) -> Result<
        eredu_runtime::replicated_session::PrefillSourceProgress<Option<ReferenceTensor>>,
        Error,
    >;
    fn verify(
        &mut self,
        tokens: &ReferenceTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
    ) -> Result<
        (
            ReferenceTensor,
            ExternalPredictionTargetCapture<ReferenceTensor>,
        ),
        Error,
    >;
    fn operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, ReferenceTensor>,
    ) -> Result<ReferenceTensor, Error>;
}

struct ConstructedReferenceTarget<A>
where
    A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error> + 'static,
{
    session: ReplicatedTextSession<
        PreparedCompositeArchitecture<A>,
        ReferenceBackend,
        ReferenceReplicatedMechanisms,
    >,
    admission: A::AdmissionConfig,
}

impl<A> ConstructedReferenceTarget<A>
where
    A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
{
    fn with_lane<T>(
        &mut self,
        cache: &mut ReferenceState,
        operation: impl FnOnce(
            &mut ReplicatedTextSession<
                PreparedCompositeArchitecture<A>,
                ReferenceBackend,
                ReferenceReplicatedMechanisms,
            >,
        ) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.session
            .exchange_prediction_target_state(cache, &())
            .map_err(|error| Error::backend(error.to_string()))?;
        let output = operation(&mut self.session);
        if let Err(error) = self.session.exchange_prediction_target_state(cache, &()) {
            self.session
                .recover_prediction_target_state_after_failure(cache)
                .map_err(|restore| {
                    Error::backend(format!(
                        "reference target state restore failed after {error}: {restore}"
                    ))
                })?;
            return Err(Error::backend(error.to_string()));
        }
        output
    }

    fn run_capture(
        &mut self,
        input: ReferencePreparedInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
        prefill: bool,
    ) -> Result<
        (
            ReferenceTensor,
            ExternalPredictionTargetCapture<ReferenceTensor>,
        ),
        Error,
    > {
        let admitted =
            A::admit_prepared_input(&self.admission, &input.input, &ReferenceInputInspector)
                .map_err(|error| Error::backend(error.to_string()))?;
        let paired =
            PreparedCompositeInput::new(&input.input, &admitted).map_err(Error::backend)?;
        let paths = A::external_prediction_capture_paths(request)?
            .ok_or_else(|| Error::backend("capture request differs from reference target"))?;
        let mut observer = ExactReferenceCaptureObserver::new(paths);
        let captured = observer.values.clone();
        let request = request.clone();
        let identity = input.identity;
        self.with_lane(cache, |session| {
            let capture = |forward: &A::ForwardContext| {
                let values = captured
                    .borrow()
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, value)| {
                        value.ok_or_else(|| {
                            Error::backend(format!("reference target missed capture path {index}"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                A::external_prediction_capture(&request, forward, values)?
                    .ok_or_else(|| Error::backend("target did not form selected capture"))
            };
            let result = if prefill {
                session.prefill_input_with_capture(paired, &(), &mut observer, capture)
            } else {
                session.decode_input_with_capture(paired, &(), &mut observer, capture)
            }
            .map_err(|error| Error::backend(error.to_string()))?;
            let _ = identity;
            Ok(result)
        })
    }
}

impl<A> ReferenceExternalTarget for ConstructedReferenceTarget<A>
where
    A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error> + 'static,
    A::InputPartPlan: 'static,
    A::Error: std::fmt::Display,
{
    fn profile(&self) -> eredu_architectures::external_assistant::ExternalAssistantTargetProfile {
        A::external_assistant_target_profile(&self.admission)
            .expect("construction dispatch admitted external prediction")
    }

    fn prepare_cache(&mut self) -> Result<ReferenceState, Error> {
        self.session
            .prepare_prediction_target_state(&())
            .map_err(|error| Error::backend(error.to_string()))
    }

    fn prefill(
        &mut self,
        input: ReferencePreparedInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
    ) -> Result<
        (
            ReferenceTensor,
            ExternalPredictionTargetCapture<ReferenceTensor>,
        ),
        Error,
    > {
        self.run_capture(input, request, cache, true)
    }

    fn prefill_spans(
        &mut self,
        input: ReferencePreparedInput,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
            ReferenceTensor,
            Error,
        >,
        cancellation: &GenerationCancellationToken,
        origin: eredu_core::speculative::SpeculativeActivationOrigin,
    ) -> Result<
        eredu_runtime::replicated_session::PrefillSourceProgress<Option<ReferenceTensor>>,
        Error,
    > {
        let admitted =
            A::admit_prepared_input(&self.admission, &input.input, &ReferenceInputInspector)
                .map_err(|e| Error::backend(e.to_string()))?;
        let shape = admitted.decoder_shape();
        let allow_whole_input = input.chunk.is_none();
        let chunk = receiver.prefill_chunk_positions(
            input.chunk,
            Some(shape[1]),
            eredu_architectures::prefill::is_prepared_token_input(&input.input),
        );
        let admission = self.admission.clone();
        let cached_positions = u64::try_from(eredu_nn::AttentionCache::offset(
            eredu_runtime::LayerRuntimeState::layer(cache, 0).map_err(Error::backend)?,
        ))
        .map_err(Error::backend)?;
        self.with_lane(cache, |session| {
            let (selected, shape_kind, config) = input.external_schedule.as_ref()
                .ok_or_else(|| Error::backend("reference external input has no selected schedule"))?;
            let (authority, original) = reference_external_schedule(
                selected, *shape_kind, config,
                session.inference_execution_identity(),
                eredu_core::InferenceGeometry {
                    batch_size: shape[0],
                    cached_positions,
                    input_positions: shape[1],
                    max_output_tokens: u64::try_from(config.max_tokens).map_err(Error::backend)?,
                    prefill_chunk_positions: chunk
                        .map_or(eredu_runtime::prefill::DEFAULT_PREFILL_CHUNK_POSITIONS,
                            |value| value.get()).min(shape[1]),
                    output: receiver.output_demand(),
                },
            )
            .map_err(Error::backend)?;
            session
                .try_prefill_speculative_source_with_operation(
                    authority,
                    Some(shape),
                    chunk,
                    receiver.output_demand(),
                    |geometry| {
                        eredu_architectures::prefill::PreparedExternalPrefill::from_prepared(
                            input.input,
                            admitted,
                            geometry,
                            admission,
                            ReferenceInputInspector,
                            allow_whole_input,
                        )
                    },
                    cancellation,
                    &(),
                    &mut eredu_runtime::NoopObserver,
                    ReferenceExternalSpan::<A> {
                        request,
                        receiver,
                        original,
                        origin,
                        _architecture: std::marker::PhantomData,
                    },
                )
                .map_err(|error| Error::backend(error.to_string()))
        })
    }

    fn verify(
        &mut self,
        tokens: &ReferenceTensor,
        request: &ExternalPredictionCaptureRequest,
        cache: &mut ReferenceState,
    ) -> Result<
        (
            ReferenceTensor,
            ExternalPredictionTargetCapture<ReferenceTensor>,
        ),
        Error,
    > {
        let sequence = usize::try_from(tokens.dim(1)).map_err(Error::backend)?;
        self.run_capture(
            ReferencePreparedInput::tokens(&vec![0; sequence])?,
            request,
            cache,
            false,
        )
    }

    fn operation(
        &mut self,
        operation: ExternalPredictionTargetOperation<'_, ReferenceTensor>,
    ) -> Result<ReferenceTensor, Error> {
        self.session
            .apply_prediction_target_operation(ReferenceCompositeOperation { operation }, &())
            .map_err(|error| Error::backend(error.to_string()))
    }
}

struct ConstructReferenceTargetVisitor {
    construction_started: bool,
}

impl
    eredu_architectures::replicated_text::CompositeTextArchitectureVisitor<
        ReferenceBackend,
        ReferenceState,
    > for ConstructReferenceTargetVisitor
{
    type Output = Box<dyn ReferenceExternalTarget>;
    type Error = Error;

    fn construction_started(&mut self) {
        self.construction_started = true;
    }

    fn visit<A>(
        self,
        prepared: eredu_architectures::replicated_text::PreparedCompositeTextArchitecture<
            A,
            A::AdmissionConfig,
        >,
        _: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<ReferenceBackend, ReferenceState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        assert!(self.construction_started);
        let (architecture, source, contract, _, admission) = prepared.into_parts();
        let session = eredu_runtime::construct_replicated_text_session::<_, ReferenceBackend, _>(
            architecture,
            source,
            contract,
            ReferenceReplicatedMechanisms,
            &(),
        )
        .map_err(|error| Error::backend(error.to_string()))?;
        Ok(Box::new(ConstructedReferenceTarget { session, admission }))
    }

    fn visit_routed<A>(
        self,
        _: eredu_architectures::replicated_text::PreparedRoutedCompositeTextArchitecture<
            A,
            A::AdmissionConfig,
        >,
        _: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: CompositeArchitecture<ReferenceBackend, ReferenceState, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<ReferenceBackend, ReferenceState>
            + 'static,
        A::InputPartPlan: 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        Err(Error::backend(
            "reference production proof selected an unexpected routed target",
        ))
    }
}

fn reference_composite_selection(
    requirements: &eredu_architectures::replicated_text::CompositeTextRequirements,
    input: &eredu_runtime::PreparedModelInput<ReferenceTensor>,
) -> eredu_architectures::replicated_text::SelectedCompositeTextRealization {
    // These shape-only operators are eager, and ReferenceReplicatedMechanisms
    // completes their actual state synchronously.
    let capabilities =
        reference_text_capabilities(requirements.execution()).with_exact_completion(true);
    let processor_request = eredu_runtime::ProcessorSelectionRequest::new(
        input.parts().iter().map(|part| part.modality()),
    )
    .with_prepared_tensors(true);
    let processor_capabilities = eredu_runtime::MediaPrimitiveCapabilities::new(
        [],
        [
            eredu_core::InputModality::Text,
            eredu_core::InputModality::Image,
            eredu_core::InputModality::Video,
            eredu_core::InputModality::Audio,
        ],
        [
            eredu_core::InputModality::Text,
            eredu_core::InputModality::Image,
            eredu_core::InputModality::Video,
            eredu_core::InputModality::Audio,
        ],
        [],
        i32::MAX as u64,
    );
    eredu_architectures::replicated_text::select_composite_text_realization(
        requirements,
        &eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        ),
        eredu_runtime::WeightResidency::fully_resident(),
        &processor_request,
        &capabilities,
        &processor_capabilities,
    )
    .unwrap()
}

fn reference_text_capabilities(
    execution: &eredu_runtime::ReplicatedTextRequirements,
) -> eredu_runtime::BackendMechanismCapabilities {
    let lowerings = execution
        .parameters()
        .iter()
        .chain(execution.auxiliary_parameters())
        .filter(|parameter| parameter.has_lowering_source())
        .map(|parameter| {
            let kind = if matches!(
                parameter.presence(),
                eredu_runtime::ReplicatedTextParameterPresence::Derived { .. }
            ) {
                eredu_runtime::WeightLoweringKind::Derived
            } else {
                eredu_runtime::WeightLoweringKind::Direct
            };
            eredu_runtime::WeightLoweringCapability::new(
                parameter
                    .lowering_descriptor(parameter.native_executable())
                    .unwrap(),
                kind,
            )
        })
        .collect();
    let state = eredu_runtime::StateMechanismCapabilities::new(
        (0..execution.state_layout().len()).flat_map(|layer| {
            execution
                .state_layout()
                .components(layer)
                .unwrap()
                .iter()
                .cloned()
                .map(move |component| {
                    eredu_runtime::StateComponentMechanism::new(
                        layer,
                        component,
                        Some(eredu_runtime::StateComponentPlacement::Device),
                        None,
                    )
                })
        }),
    )
    .with_floating_state_dtype(
        execution.floating_state_source().unwrap().clone(),
        eredu_runtime::StateStorageDtype::F32,
    )
    .with_transactions(true, true)
    .with_reset(true);
    eredu_runtime::BackendMechanismCapabilities::new(
        eredu_nn::NeuralOperatorCapabilities::ALL,
        lowerings,
        vec![eredu_runtime::WeightResidencyMechanism::Resident],
        state,
    )
    .with_grouped_operations([
        eredu_runtime::GroupedOperationRequirement::GatedProduct,
        eredu_runtime::GroupedOperationRequirement::GatedProductTensorParallelPartial,
    ])
}

fn reference_composite_artifact(config: &serde_json::Value) -> tempfile::TempDir {
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(
        artifact.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(config)
        .unwrap();
    let checkpoint = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let mut constraints = checkpoint.common_tensors.iter().collect::<Vec<_>>();
    constraints.extend(
        checkpoint
            .layout_groups
            .iter()
            .filter(|group| group.required)
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| variant.tensors.iter()),
    );
    let mut tensors = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for constraint in constraints.into_iter().filter(|constraint| {
        constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
    }) {
        tensors
            .entry(constraint.key.clone())
            .or_insert_with(|| constraint.shape.clone());
    }
    write_sparse_safetensors(&artifact.path().join("model.safetensors"), &tensors);
    artifact
}

fn write_sparse_safetensors(
    path: &std::path::Path,
    tensors: &std::collections::BTreeMap<String, Vec<usize>>,
) {
    use std::io::Write as _;

    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    for (name, shape) in tensors {
        let bytes = shape
            .iter()
            .try_fold(4_u64, |bytes, dimension| {
                bytes.checked_mul(u64::try_from(*dimension).ok()?)
            })
            .expect("reference sparse tensor size must fit u64");
        let end = offset
            .checked_add(bytes)
            .expect("reference sparse checkpoint offset must fit u64");
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype":"F32",
                "shape":shape,
                "data_offsets":[offset,end]
            }),
        );
        offset = end;
    }
    let mut encoded = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !encoded.len().is_multiple_of(8) {
        encoded.push(b' ');
    }
    let header_len = u64::try_from(encoded.len()).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&header_len.to_le_bytes()).unwrap();
    file.write_all(&encoded).unwrap();
    file.set_len(8 + header_len + offset).unwrap();
}

fn construct_reference_composite_target(
    config: &serde_json::Value,
) -> (Box<dyn ReferenceExternalTarget>, tempfile::TempDir) {
    let artifact = reference_composite_artifact(config);
    let inspection = eredu_architectures::configuration::inspect_artifact(artifact.path()).unwrap();
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection).unwrap();
    let input = ReferencePreparedInput::tokens(&[9]).unwrap();
    let selected = reference_composite_selection(&requirements, &input.input);
    let store: eredu_checkpoint::store::RetainedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    )
    .into();
    let target = eredu_architectures::replicated_text::visit_composite_text_architecture::<
        ReferenceBackend,
        ReferenceState,
        _,
    >(
        requirements,
        selected,
        store,
        &(),
        ConstructReferenceTargetVisitor {
            construction_started: false,
        },
    )
    .unwrap();
    (target, artifact)
}

struct ReferenceMaterializedAssistant<A: eredu_architectures::ExternalAssistantArchitecture> {
    config: A::Config,
    module: A::Module<ReferenceBackend>,
    observers: eredu_architectures::external_assistant::ExternalAssistantObservers<
        ReferenceTensor,
        u32,
        Error,
    >,
}

struct ReferenceAssistantMaterializer;

impl eredu_architectures::ExternalAssistantPreparationVisitor for ReferenceAssistantMaterializer {
    type Output<A: eredu_architectures::ExternalAssistantArchitecture> =
        ReferenceMaterializedAssistant<A>;
    type Error = Error;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        prepared: eredu_architectures::PreparedExternalAssistantSource<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        let (store, checkpoint, _identity, _source_config, config, _tasks) = prepared.into_parts();
        if matches!(
            checkpoint,
            eredu_architectures::ExternalAssistantCheckpoint::Gguf { .. }
        ) {
            return Err(Error::backend(
                "reference production proof requires its SafeTensors fixture",
            ));
        }
        let mut module = A::module::<ReferenceBackend>(config.clone(), &())?;
        struct Bindings(Vec<eredu_runtime::WeightBinding>);
        impl<'a> ParameterVisitor<'a, ReferenceTensor> for Bindings {
            fn visit(
                &mut self,
                metadata: eredu_nn::ParameterMetadataView<'_>,
                value: &'a ReferenceTensor,
            ) {
                let expected_bytes = value
                    .shape()
                    .iter()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .product::<u64>()
                    * 4;
                self.0.push(
                    // ReferenceBackend materializes shapes. Keep the exact source
                    // recipe and binding checks without reading the released-size
                    // sparse payload which this fixture never uses numerically.
                    eredu_runtime::WeightBinding::from_recipe(
                        metadata.id().as_str(),
                        eredu_checkpoint::recipe::DerivedWeightRecipe::source(
                            metadata.id().as_str(),
                            eredu_checkpoint::store::TensorSelection::Full,
                        ),
                        expected_bytes,
                    )
                    .unwrap(),
                );
            }
        }
        let mut bindings = Bindings(Vec::new());
        module.visit_parameters(&mut bindings).map_err(Error::backend)?;
        let materialized = eredu_runtime::materialize_bindings::<ReferenceBackend>(
            store.as_ref(),
            &bindings.0,
            &(),
        )
        .map_err(|error| Error::backend(error.to_string()))?;
        eredu_runtime::bind_materialized_unit::<ReferenceBackend, _>(&mut module, materialized)
            .map_err(|error| Error::backend(error.to_string()))?;
        Ok(ReferenceMaterializedAssistant {
            config,
            module,
            observers: Default::default(),
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum ReferenceCompletionMode {
    Immediate,
    Delayed { incomplete_polls: usize },
    FailWait,
    Never,
}

#[derive(Debug, Default)]
struct ReferenceCompletionControl {
    submissions: Cell<usize>,
    polls: Cell<usize>,
    incomplete_polls: Cell<usize>,
    waits: Cell<usize>,
    failures: Cell<usize>,
    quarantines: Cell<usize>,
    drops: Cell<usize>,
    retained_at_incomplete_poll: Cell<usize>,
    retained_at_wait: Cell<usize>,
    released_resources: Cell<usize>,
    publications: Cell<usize>,
    restores: Cell<usize>,
    exact_restores: Cell<usize>,
    lifecycle: RefCell<Vec<eredu_core::SpeculativeLifecycleStage>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ReferenceCompletionEvidence {
    submissions: usize,
    polls: usize,
    incomplete_polls: usize,
    waits: usize,
    failures: usize,
    quarantines: usize,
    drops: usize,
    retained_at_incomplete_poll: usize,
    retained_at_wait: usize,
    released_resources: usize,
    publications: usize,
    restores: usize,
    exact_restores: usize,
    lifecycle: Vec<eredu_core::SpeculativeLifecycleStage>,
}

impl ReferenceCompletionControl {
    fn evidence(&self) -> ReferenceCompletionEvidence {
        ReferenceCompletionEvidence {
            submissions: self.submissions.get(),
            polls: self.polls.get(),
            incomplete_polls: self.incomplete_polls.get(),
            waits: self.waits.get(),
            failures: self.failures.get(),
            quarantines: self.quarantines.get(),
            drops: self.drops.get(),
            retained_at_incomplete_poll: self.retained_at_incomplete_poll.get(),
            retained_at_wait: self.retained_at_wait.get(),
            released_resources: self.released_resources.get(),
            publications: self.publications.get(),
            restores: self.restores.get(),
            exact_restores: self.exact_restores.get(),
            lifecycle: self.lifecycle.borrow().clone(),
        }
    }
}

#[derive(Clone)]
struct ReferenceCompletionInjection {
    mode: ReferenceCompletionMode,
    control: Rc<ReferenceCompletionControl>,
    before_publication: Option<Rc<ReferenceCompletionControl>>,
}

thread_local! {
    static REFERENCE_COMPLETION_CONTROL: RefCell<Option<ReferenceCompletionInjection>> =
        const { RefCell::new(None) };
    static REFERENCE_COMPLETION_QUARANTINE: RefCell<ReferenceCompletionQuarantine> =
        RefCell::new(ReferenceCompletionQuarantine::default());
}

#[derive(Debug, Default)]
struct ReferenceCompletionQuarantine {
    work: Vec<ReferenceExternalCompletion>,
}

impl Drop for ReferenceCompletionQuarantine {
    fn drop(&mut self) {
        for completion in self.work.drain(..) {
            // A genuinely never-completing operation has no safe release point. The
            // reference backend deliberately retains its synthetic resources forever.
            std::mem::forget(completion);
        }
    }
}

fn with_reference_completion_control<R>(
    mode: ReferenceCompletionMode,
    operation: impl FnOnce() -> R,
) -> (R, ReferenceCompletionEvidence) {
    let control = Rc::new(ReferenceCompletionControl::default());
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "reference completion control is nested"
        );
        *slot.borrow_mut() = Some(ReferenceCompletionInjection {
            mode,
            control: Rc::clone(&control),
            before_publication: None,
        });
    });
    let result = operation();
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        slot.borrow_mut().take();
    });
    (result, control.evidence())
}

// Verification injection starts only after the actual Publisher has committed
// the target-only token. Capture/seed completions retain separate evidence.
fn with_reference_verification_completion_control<R>(
    mode: ReferenceCompletionMode,
    operation: impl FnOnce() -> R,
) -> (R, ReferenceCompletionEvidence, ReferenceCompletionEvidence) {
    let control = Rc::new(ReferenceCompletionControl::default());
    let prefill = Rc::new(ReferenceCompletionControl::default());
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "reference completion control is nested"
        );
        *slot.borrow_mut() = Some(ReferenceCompletionInjection {
            mode,
            control: Rc::clone(&control),
            before_publication: Some(Rc::clone(&prefill)),
        });
    });
    let result = operation();
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        slot.borrow_mut().take();
    });
    (result, control.evidence(), prefill.evidence())
}

fn record_reference_publication() {
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        if let Some(injection) = slot.borrow().as_ref() {
            let control = &injection.control;
            if control.publications.get() == 0 {
                if let Some(prefill) = &injection.before_publication {
                    assert_eq!(
                        prefill.submissions.get(),
                        prefill.drops.get(),
                        "capture/seed completion must retire before target publication"
                    );
                    assert_eq!(
                        prefill.retained_at_wait.get(),
                        prefill.released_resources.get(),
                        "capture/seed roots must retire before target publication"
                    );
                    assert_eq!(prefill.failures.get(), 0);
                }
            }
            control.publications.set(control.publications.get() + 1);
        }
    });
}

fn record_reference_lifecycle(stage: eredu_core::SpeculativeLifecycleStage) {
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        if let Some(injection) = slot.borrow().as_ref() {
            let control = &injection.control;
            control.lifecycle.borrow_mut().push(stage);
        }
    });
}

fn record_reference_restore(exact: bool) {
    REFERENCE_COMPLETION_CONTROL.with(|slot| {
        if let Some(injection) = slot.borrow().as_ref() {
            let control = &injection.control;
            control.restores.set(control.restores.get() + 1);
            if exact {
                control.exact_restores.set(control.exact_restores.get() + 1);
            }
        }
    });
}

#[derive(Debug)]
struct ReferenceExternalCompletion {
    mode: ReferenceCompletionMode,
    remaining_incomplete_polls: Cell<usize>,
    retained: Vec<ReferenceTensor>,
    control: Option<Rc<ReferenceCompletionControl>>,
    _original_role: Option<eredu_runtime::working_memory::OriginalExternalSpeculativeRole>,
}

impl ReferenceExternalCompletion {
    fn submit(retained: impl IntoIterator<Item = ReferenceTensor>) -> Self {
        let retained = retained.into_iter().collect::<Vec<_>>();
        let configured = REFERENCE_COMPLETION_CONTROL.with(|slot| slot.borrow().clone());
        let (mode, control) = configured
            .map(|injection| {
                if injection.control.publications.get() == 0 {
                    if let Some(prefill) = injection.before_publication {
                        return (ReferenceCompletionMode::Immediate, Some(prefill));
                    }
                }
                (injection.mode, Some(injection.control))
            })
            .unwrap_or((ReferenceCompletionMode::Immediate, None));
        if let Some(control) = &control {
            control.submissions.set(control.submissions.get() + 1);
        }
        let remaining_incomplete_polls = match mode {
            ReferenceCompletionMode::Delayed { incomplete_polls } => incomplete_polls,
            ReferenceCompletionMode::Never => usize::MAX,
            ReferenceCompletionMode::Immediate | ReferenceCompletionMode::FailWait => 0,
        };
        Self {
            mode,
            remaining_incomplete_polls: Cell::new(remaining_incomplete_polls),
            retained,
            control,
            _original_role: REFERENCE_SPAN_ROLE.with(|role| role.borrow().clone()),
        }
    }
}

impl eredu_core::Completion for ReferenceExternalCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        if let Some(control) = &self.control {
            control.polls.set(control.polls.get() + 1);
        }
        if matches!(self.mode, ReferenceCompletionMode::Never) {
            if let Some(control) = &self.control {
                control
                    .incomplete_polls
                    .set(control.incomplete_polls.get() + 1);
                control
                    .retained_at_incomplete_poll
                    .set(control.retained_at_incomplete_poll.get() + self.retained.len());
            }
            return Ok(false);
        }
        let remaining = self.remaining_incomplete_polls.get();
        if remaining > 0 {
            self.remaining_incomplete_polls.set(remaining - 1);
            if let Some(control) = &self.control {
                control
                    .incomplete_polls
                    .set(control.incomplete_polls.get() + 1);
                control
                    .retained_at_incomplete_poll
                    .set(control.retained_at_incomplete_poll.get() + self.retained.len());
            }
            return Ok(false);
        }
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        if let Some(control) = &self.control {
            control.waits.set(control.waits.get() + 1);
            control
                .retained_at_wait
                .set(control.retained_at_wait.get() + self.retained.len());
        }
        if matches!(
            self.mode,
            ReferenceCompletionMode::FailWait | ReferenceCompletionMode::Never
        ) {
            if let Some(control) = &self.control {
                control.failures.set(control.failures.get() + 1);
            }
            return Err(Error::backend(match self.mode {
                ReferenceCompletionMode::Never => {
                    "never-completing reference completion was quarantined"
                }
                _ => "injected reference exact-completion failure",
            }));
        }
        Ok(())
    }
}

impl eredu_core::BoundedCompletion for ReferenceExternalCompletion {
    fn wait_bounded(
        self,
        policy: eredu_core::BoundedCompletionWait,
    ) -> Result<eredu_core::BoundedCompletionOutcome, Self::Error> {
        if !matches!(self.mode, ReferenceCompletionMode::Never) {
            self.wait()?;
            return Ok(eredu_core::BoundedCompletionOutcome::Completed);
        }
        let deadline = std::time::Instant::now() + policy.timeout();
        while std::time::Instant::now() < deadline {
            assert!(!self.is_complete()?);
            std::thread::yield_now();
        }
        if let Some(control) = &self.control {
            control.quarantines.set(control.quarantines.get() + 1);
        }
        REFERENCE_COMPLETION_QUARANTINE.with(|quarantine| {
            quarantine.borrow_mut().work.push(self);
        });
        Ok(eredu_core::BoundedCompletionOutcome::DeadlineExceeded {
            cancellation: eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        })
    }
}

impl Drop for ReferenceExternalCompletion {
    fn drop(&mut self) {
        if let Some(control) = &self.control {
            control.drops.set(control.drops.get() + 1);
            control
                .released_resources
                .set(control.released_resources.get() + self.retained.len());
        }
    }
}

#[derive(Clone, Copy, Default)]
struct ReferenceExternalContext {
    origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>,
}
struct ReferenceExternalMechanisms;

impl<A> eredu_architectures::ExternalAssistantExecutionMechanisms<A> for ReferenceExternalMechanisms
where
    A: eredu_architectures::ExternalAssistantArchitecture,
{
    type NeuralBackend = ReferenceBackend;
    type AttentionCache = ReferenceCache;
    type Target = dyn ReferenceExternalTarget;
    type Assistant = ReferenceMaterializedAssistant<A>;
    type Input = ReferencePreparedInput;
    type NativeCache = ReferenceState;
    type NativeCacheCheckpoint = ReferenceState;
    type Tensor = ReferenceTensor;
    type Logits = u32;
    type Context<'a> = ReferenceExternalContext;
    fn requires_activation_origin() -> bool { true }
    fn invocation_context<'a>(_: Self::Context<'a>, origin: Option<eredu_core::speculative::SpeculativeActivationOrigin>) -> Result<Self::Context<'a>, Error> {
        Ok(ReferenceExternalContext { origin })
    }
    type Completion = ReferenceExternalCompletion;
    type Telemetry = ();
    type Error = Error;

    fn source_context<'a, 'scope>(
        context: Self::Context<'a>,
        _sources: &'scope [&'scope eredu_architectures::speculative_execution::PreparedEmbeddedEvidence],
    ) -> Result<Self::Context<'scope>, Self::Error>
    where
        'a: 'scope,
    {
        Ok(context)
    }

    fn config(assistant: &Self::Assistant) -> &A::Config {
        &assistant.config
    }

    fn module(assistant: &mut Self::Assistant) -> &mut A::Module<Self::NeuralBackend> {
        &mut assistant.module
    }

    fn neural_error(error: Error) -> Self::Error {
        error
    }

    fn error(message: String) -> Self::Error {
        Error::backend(message)
    }

    fn prepared_input_cache_identity(
        input: &Self::Input,
    ) -> Result<eredu_runtime::PreparedInputCacheIdentity, Self::Error> {
        Ok(input.identity.clone())
    }

    fn tensor_shape(value: &Self::Tensor) -> Result<Vec<usize>, Self::Error> {
        value
            .shape()
            .iter()
            .map(|dimension| usize::try_from(*dimension).map_err(Error::backend))
            .collect()
    }

    fn prefill_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        _: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        target.prefill(input, request, cache)
    }

    fn prefill_chunk_positions(input: &Self::Input) -> Option<std::num::NonZeroU64> {
        input.chunk
    }
    fn prefill_target_spans_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        input: Self::Input,
        cache: &mut Self::NativeCache,
        receiver: &mut dyn eredu_architectures::external_assistant::ExternalPrefillReceiver<
            Self::Tensor,
            Self::Error,
        >,
        cancellation: &GenerationCancellationToken,
        context: Self::Context<'a>,
    ) -> Result<
        eredu_runtime::replicated_session::PrefillSourceProgress<Option<Self::Tensor>>,
        Self::Error,
    > {
        target.prefill_spans(input, request, cache, receiver, cancellation, context.origin.ok_or_else(|| Error::backend("missing scheduler origin"))?)
    }
    fn supports_prefill_observation(assistant: &Self::Assistant, context: bool) -> bool {
        assistant.observers.supports_prefill(context)
    }
    fn prefill_output_demand(assistant: &Self::Assistant) -> eredu_core::OutputDemand {
        assistant.observers.prefill_output_demand()
    }
    fn begin_prefill_chunk(
        assistant: &mut Self::Assistant,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), Self::Error> {
        assistant.observers.begin_prefill_chunk(chunk)
    }
    fn begin_prefill_context(
        assistant: &mut Self::Assistant,
        frontier: u64,
    ) -> Result<(), Self::Error> {
        assistant.observers.begin_prefill_context(frontier)
    }
    fn finish_prefill(assistant: &mut Self::Assistant, committed: bool) {
        assistant.observers.finish_prefill(committed);
    }

    fn verify_target_native<'a>(
        target: &mut Self::Target,
        request: &ExternalPredictionCaptureRequest,
        tokens: &Self::Tensor,
        cache: &mut Self::NativeCache,
        _: Self::Context<'a>,
    ) -> Result<(Self::Tensor, ExternalPredictionTargetCapture<Self::Tensor>), Self::Error> {
        target.verify(tokens, request, cache)
    }

    fn checkpoint_native(
        cache: &Self::NativeCache,
    ) -> Result<Self::NativeCacheCheckpoint, Self::Error> {
        Ok(cache.clone())
    }

    fn restore_checkpoint_native<'a>(
        cache: &mut Self::NativeCache,
        checkpoint: &Self::NativeCacheCheckpoint,
        _: Self::Context<'a>,
    ) -> Result<(), Self::Error> {
        cache.clone_from(checkpoint);
        let exact = cache.as_ref().len() == checkpoint.as_ref().len()
            && cache
                .as_ref()
                .iter()
                .zip(checkpoint.as_ref())
                .all(|(actual, expected)| {
                    actual.offset == expected.offset
                        && actual.window == expected.window
                        && actual.resets == expected.resets
                        && actual.fixed.as_ref().map(ReferenceTensor::shape)
                            == expected.fixed.as_ref().map(ReferenceTensor::shape)
                });
        record_reference_restore(exact);
        Ok(())
    }

    fn native_cache_len(cache: &Self::NativeCache) -> Result<i32, Self::Error> {
        cache
            .as_ref()
            .first()
            .map(RuntimeStateComponents::position)
            .ok_or_else(|| Error::backend("reference target cache has no layers"))
    }

    fn observe_tensor(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        assistant.observers.observe_tensor(path, &value)
    }

    fn observe_logits(
        assistant: &mut Self::Assistant,
        path: &str,
        value: Self::Logits,
    ) -> Result<Self::Logits, Self::Error> {
        assistant.observers.observe_logits(path, &value)
    }

    fn sequence_len(value: &Self::Tensor) -> Result<usize, Self::Error> {
        value
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Error::backend("reference sequence tensor has rank below two"))
            .and_then(|length| usize::try_from(length).map_err(Error::backend))
    }

    fn sequence_row<'a>(
        value: &Self::Tensor,
        row: usize,
        retain_dimension: bool,
        _: eredu_architectures::external_assistant::ExternalAssistantTensorPlacement,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let length = value
            .shape()
            .get(1)
            .copied()
            .ok_or_else(|| Error::backend("reference sequence tensor has rank below two"))
            .and_then(|length| usize::try_from(length).map_err(Error::backend))?;
        if row >= length {
            return Err(Error::backend("reference sequence row is out of bounds"));
        }
        let mut shape = value.shape().to_vec();
        if retain_dimension {
            shape[1] = 1;
        } else {
            shape.remove(1);
        }
        Ok(ReferenceTensor(shape))
    }

    fn into_logits(_: Self::Tensor) -> Self::Logits {
        2
    }

    fn sequence_suffix<'a>(
        value: &Self::Tensor,
        maximum: i32,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut shape = value.shape().to_vec();
        shape[1] = shape[1].min(maximum);
        Ok(ReferenceTensor(shape))
    }

    fn shared_prefix<'a>(
        value: &Self::Tensor,
        cache_len: i32,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut shape = value.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::backend(
                "reference shared attention tensor is not rank four",
            ));
        }
        shape[2] = shape[2].min(cache_len);
        Ok(ReferenceTensor(shape))
    }

    fn token_prefix<'a>(
        value: &Self::Tensor,
        end: usize,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        let mut shape = value.shape().to_vec();
        shape[1] = i32::try_from(end).map_err(Error::backend)?;
        Ok(ReferenceTensor(shape))
    }

    fn target_tokens<'a>(
        tokens: &[u32],
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            1,
            i32::try_from(tokens.len()).map_err(Error::backend)?,
        ]))
    }

    fn transfer<'a>(
        value: &Self::Tensor,
        _: eredu_architectures::external_assistant::ExternalAssistantTransfer,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(value.clone())
    }

    fn target_operation<'a>(
        target: &mut Self::Target,
        operation: ExternalPredictionTargetOperation<'_, Self::Tensor>,
        _: Self::Context<'a>,
    ) -> Result<Self::Tensor, Self::Error> {
        target.operation(operation)
    }

    fn neural_context<'a>(
        _: Self::Context<'a>,
        _: eredu_architectures::external_assistant::ExternalAssistantTensorPlacement,
    ) -> &'a () {
        &()
    }

    fn submit_completion<'a>(
        values: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a,
    {
        Ok(ReferenceExternalCompletion::submit(
            values.into_iter().cloned(),
        ))
    }
}

fn reference_gemma_assistant_artifact() -> tempfile::TempDir {
    const CONFIG: &str = r#"{
      "model_type":"gemma4_assistant","backbone_hidden_size":32,
      "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":3,
      "text_config":{"model_type":"gemma4_text","hidden_size":32,
        "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
        "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
        "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
        "attention_k_eq_v":false,"layer_types":["full_attention"]}
    }"#;
    let config =
        eredu_architectures::gemma4::AssistantConfig::from_json(CONFIG.as_bytes()).unwrap();
    let plan = eredu_architectures::gemma4::assistant_safetensors_plan(&config).unwrap();
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| (tensor.key.clone(), tensor.shape.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(artifact.path().join("config.json"), CONFIG).unwrap();
    write_sparse_safetensors(&artifact.path().join("model.safetensors"), &tensors);
    artifact
}

fn reference_muse_assistant_artifact() -> tempfile::TempDir {
    const CONFIG: &str = r#"{
      "model_type":"muse_glimmer_assistant","hidden_size":6656,
      "intermediate_size":19968,"num_hidden_layers":5,"num_attention_heads":32,
      "num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
      "max_position_embeddings":131072,"sliding_window":2048,"block_size":16,
      "mask_token_id":201818,"target_layer_ids":[1,13,25,37,49],
      "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                     "sliding_attention","sliding_attention"],
      "hidden_act":"silu","attention_dropout":0.0,
      "rope_parameters":{"rope_theta":500000.0}
    }"#;
    let config =
        eredu_architectures::muse_glimmer::DFlashConfig::from_hf_json(CONFIG.as_bytes()).unwrap();
    let plan = eredu_architectures::muse_glimmer::dflash_safetensors_plan(&config).unwrap();
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| (tensor.key.clone(), tensor.shape.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    let artifact = tempfile::tempdir().unwrap();
    std::fs::write(artifact.path().join("config.json"), CONFIG).unwrap();
    write_sparse_safetensors(&artifact.path().join("model.safetensors"), &tensors);
    artifact
}

#[derive(Debug)]
struct ReferenceProductionOutcome {
    tokens: Vec<u32>,
    accepted: Vec<usize>,
    target_tokens: usize,
    publications: usize,
    construction_stages: Vec<eredu_core::SpeculativeLifecycleStage>,
    execution_stages: Vec<eredu_core::SpeculativeLifecycleStage>,
}

fn reference_embedded_schedule(
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    shape: eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape,
    alignment: eredu_core::speculative::PredictionPrefillAlignment,
    geometry: eredu_core::InferenceGeometry,
    config: &SpeculativeConfig,
) -> Result<eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority, String> {
    use eredu_runtime::working_memory::OriginalSpeculativeRequest;
    let plan = eredu_runtime::speculative::embedded_occurrence::EmbeddedSchedulePlan::new(
        selected,
        shape,
        alignment,
        0,
        eredu_runtime::prefill::PrefillControlPlan::new(geometry, true)
            .map_err(|e| e.to_string())?,
        geometry
            .input_positions
            .checked_add(geometry.max_output_tokens)
            .ok_or("context overflow")?,
        config,
        SpeculativeSchedulerOptions::default().with_lookahead(false),
    )
    .map_err(|e| e.to_string())?;
    let pool = crate::memory_fixture::ledger(1 << 30, 0).map_err(|e| e.to_string())?;
    let limits = eredu_core::MemoryLimits::unlimited(pool.topology());
    let request =
        OriginalSpeculativeRequest::prepare_embedded(&pool, execution, &plan, limits.clone())
            .map_err(|e| e.to_string())?;
    let funding = pool
        .prepare_workspace_metadata(execution, limits)
        .map_err(|e| e.to_string())?;
    request
        .prepare_embedded_prefill_schedule(&plan, &funding)
        .map_err(|e| e.to_string())
}

fn reference_external_schedule<'a>(
    selected: &'a eredu_runtime::SelectedSpeculativeRealization,
    shape: eredu_runtime::speculative::external_occurrence::ExternalPredictionShape,
    config: &SpeculativeConfig,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    geometry: eredu_core::InferenceGeometry,
) -> Result<(eredu_runtime::working_memory::SpeculativePrefillScheduleAuthority,
    ReferenceExternalSpanSource<'a>), String> {
    let plan = eredu_runtime::speculative::external_occurrence::ExternalSchedulePlan::new(
        selected, shape,
        eredu_runtime::prefill::PrefillControlPlan::new(geometry, true)
            .map_err(|e| e.to_string())?,
        geometry.cached_positions.checked_add(geometry.input_positions)
            .and_then(|n| n.checked_add(geometry.max_output_tokens))
            .ok_or("context overflow")?,
        config, SpeculativeSchedulerOptions::default().with_lookahead(false),
    ).map_err(|e| e.to_string())?;
    let pool = crate::memory_fixture::ledger(1 << 30, 0).map_err(|e| e.to_string())?;
    let limits = eredu_core::MemoryLimits::unlimited(pool.topology());
    let request = eredu_runtime::working_memory::OriginalSpeculativeRequest::prepare_external(
        &pool, execution, &plan, limits.clone(),
    ).map_err(|e| e.to_string())?;
    let funding = pool.prepare_workspace_metadata(execution, limits)
        .map_err(|e| e.to_string())?;
    let authority = request.prepare_external_prefill_schedule(&plan, &funding)
        .map_err(|e| e.to_string())?;
    funding.reserve_metadata(std::mem::size_of::<ReferenceExternalSpanSource<'a>>())
        .map_err(|e| e.to_string())?;
    let metadata = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(
        ReferenceShapeOnlyFacts, funding,
    ).map_err(|e| e.to_string())?;
    Ok((authority, ReferenceExternalSpanSource {
        pool: pool.clone(),
        issuer: request, cursor: plan.into_cursor(), metadata,
        placement: pool.host_placement_handle(),
    }))
}

fn run_reference_embedded_scheduler(
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    execution: &eredu_runtime::working_memory::InferenceExecutionIdentity,
    shape: eredu_runtime::speculative::embedded_occurrence::EmbeddedPredictionShape,
    alignment: eredu_core::speculative::PredictionPrefillAlignment,
    executor: &mut eredu_architectures::speculative_execution::DynEmbeddedExecutor<
        '_,
        ReferenceEmbeddedExecutorTypes,
    >,
    captured: Option<&ReferenceCapturedCase>,
) -> Result<ReferenceProductionOutcome, String> {
    let mut cache = executor.new_cache().map_err(|error| error.to_string())?;
    let publications = Rc::new(Cell::new(0));
    let execution_stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_execution = execution_stages.clone();
    let runtime = SpeculativeOutputRuntime::new(
        ReferenceProductionSampling::<ReferenceEmbeddedSamplingContext>::default(),
        GenerationSequence::new(if captured.is_some() { 13 } else { 3 }, []),
        Constraint,
        Publisher {
            publications: publications.clone(),
        },
        captured
            .map(|case| case.cancellation.clone())
            .unwrap_or_default(),
    )
    .with_lifecycle_observer(std::sync::Arc::new(move |stage| {
        observed_execution.lock().unwrap().push(stage);
        record_reference_lifecycle(stage);
        Ok(())
    }));
    let tokens = captured
        .map(|case| {
            (0..case.prompt_positions)
                .map(|i| [1, 3, 2, 4, 5][i % 5])
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![9]);
    let mut input = ReferencePreparedInput::tokens(&tokens).map_err(|error| error.to_string())?;
    input.chunk = captured.and_then(|case| case.chunk);
    let config = SpeculativeConfig {
        max_tokens: if captured.is_some() { 13 } else { 3 },
        max_draft_tokens: selected.requirements().strategy().proposal_capacity().get(),
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let schedule = reference_embedded_schedule(
        selected,
        execution,
        shape,
        alignment,
        eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: tokens.len() as u64,
            max_output_tokens: config.max_tokens as u64,
            prefill_chunk_positions: input
                .chunk
                .map_or_else(|| if captured.is_some_and(|case| !case.span_supported) {
                    tokens.len() as u64
                } else { eredu_runtime::prefill::DEFAULT_PREFILL_CHUNK_POSITIONS },
                    |n| n.get()).min(tokens.len() as u64),
            output: if captured.is_some_and(|case| case.sequence) {
                eredu_core::OutputDemand::Sequence
            } else {
                eredu_core::OutputDemand::LastPosition
            },
        },
        &config,
    )?;
    let lane = PreparedSpeculativeLane::new(
        &mut cache,
        input,
        config,
        runtime,
        SpeculativeRandomness::new(None, None),
    );
    let mut scheduler = SpeculativeScheduler::new(
        executor,
        SpeculativeSchedulerOptions::default().with_lookahead(false),
        SpeculativeExecutionTopology::Single,
        false,
        false,
        ReferenceEmbeddedContext {
            schedule: Some(&schedule),
            ..Default::default()
        },
    )
    .map_err(|error| format!("{error:?}"))?;
    scheduler.submit(lane).map_err(|error| error.to_string())?;
    scheduler.run().map_err(|error| error.to_string())?;
    let mut completed = scheduler.finish().map_err(|error| error.to_string())?;
    let request = completed
        .take_requests()
        .pop()
        .ok_or_else(|| "reference embedded scheduler returned no request".to_owned())?;
    let execution_stages = execution_stages.lock().unwrap().clone();
    Ok(ReferenceProductionOutcome {
        tokens: request.token_ids().to_vec(),
        accepted: request.stats().accept_lens().to_vec(),
        target_tokens: request.stats().target_tokens(),
        publications: publications.get(),
        construction_stages: Vec::new(),
        execution_stages,
    })
}

struct RunReferenceEmbedded {
    selected: eredu_runtime::SelectedSpeculativeRealization,
    check_snapshot_context: bool,
    captured: Option<ReferenceCapturedCase>,
}

impl
    eredu_architectures::routed_text::RoutedPredictionTargetVisitor<
        ReferenceBackend,
        ReferenceState,
        ReferencePredictionMaterializer,
    > for RunReferenceEmbedded
{
    type Output = ReferenceProductionOutcome;
    type Error = String;

    fn visit<A>(
        self,
        prepared: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        mut extension: <A as eredu_architectures::prediction_extension::MaterializedPredictionTarget<
            ReferenceBackend,
        >>::Extension<ReferencePredictionMaterializer>,
        _: eredu_checkpoint::store::RetainedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<
                ReferenceBackend,
                ReferenceState,
                Error = Error,
            > + eredu_runtime::RoutedLayeredArchitecture<ReferenceBackend, ReferenceState>
            + eredu_architectures::prediction_extension::MaterializedPredictionTarget<
                ReferenceBackend,
            > + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mut session = prepared
            .construct_resident_session(ReferenceReplicatedMechanisms, &())
            .map_err(|error| error.to_string())?;
        use eredu_architectures::prediction_extension::MaterializedPredictionExecutor;
        let execution = session.inference_execution_identity().clone();
        let shape = extension
            .occurrence_shape()
            .ok_or("missing prediction shape")?;
        let alignment = extension.prefill_alignment();
        let mut strategy =
            eredu_architectures::speculative_execution::ReplicatedMaterializedPredictionStrategy::<
                A,
                ReferenceBackend,
                ReferenceState,
                ReferenceReplicatedMechanisms,
                _,
                _,
                ReferencePredictionInput,
                ReferencePredictionMaterializer,
                ReferenceEmbeddedMechanisms,
            >::new(
                &mut session,
                &mut extension,
                &self.selected,
                ReferencePredictionInput,
                &(),
            );
        if self.check_snapshot_context {
            use eredu_architectures::speculative_execution::EmbeddedPredictionStrategy;

            let mut cache = strategy.new_cache().map_err(|error| error.to_string())?;
            let observed = RefCell::new(Vec::new());
            let context = |marker| ReferenceEmbeddedContext {
                stream: &(),
                snapshot: ReferencePredictionSnapshotContext {
                    marker,
                    observed: Some(&observed),
                },
                schedule: None,
            };
            let snapshot = strategy
                .control_target_snapshot(&cache, context(17))
                .map_err(|error| error.to_string())?
                .expect("the target and prediction snapshot must be available");
            assert!(snapshot.target().is_some());
            let target_calls = observed.take();
            assert_eq!(target_calls.first(), Some(&("target", 17)));
            assert!(target_calls.len() > 1, "prediction members must be copied");
            assert!(target_calls[1..]
                .iter()
                .all(|(member, marker)| *member != "target" && *marker == 17));

            let prediction = cache.prediction_fork().map_err(|error| error.to_string())?;
            let snapshot = strategy
                .control_prediction_snapshot(&prediction, context(29))
                .map_err(|error| error.to_string())?
                .expect("the separately retained prediction snapshot must be available");
            let prediction_calls = observed.take();
            assert_eq!(prediction_calls.len(), target_calls.len() - 1);
            assert!(prediction_calls.iter().all(|(_, marker)| *marker == 29));
            assert_eq!(
                prediction_calls
                    .iter()
                    .map(|(member, _)| member)
                    .collect::<Vec<_>>(),
                target_calls[1..]
                    .iter()
                    .map(|(member, _)| member)
                    .collect::<Vec<_>>(),
            );
            drop(snapshot);

            let target = cache.take_target().unwrap();
            assert!(strategy
                .control_target_snapshot(&cache, context(41))
                .map_err(|error| error.to_string())?
                .is_none());
            assert!(observed.borrow().is_empty());
            cache.restore_target(target);
        }
        if self.captured.as_ref().is_some_and(|case| {
            case.span_supported && case.cancel_after.is_none() && !case.cancellation.is_cancelled()
        }) {
            use eredu_architectures::speculative_execution::EmbeddedPredictionStrategy;
            let mut cache = strategy.new_cache().map_err(|error| error.to_string())?;
            let mut input = ReferencePreparedInput::tokens(&[1, 3, 2, 4, 5])
                .map_err(|error| error.to_string())?;
            input.chunk = self.captured.as_ref().and_then(|case| case.chunk);
            clear_reference_trace();
            let config = SpeculativeConfig {
                max_tokens: 13,
                max_draft_tokens: self
                    .selected
                    .requirements()
                    .strategy()
                    .proposal_capacity()
                    .get(),
                ..Default::default()
            };
            let schedule = reference_embedded_schedule(
                &self.selected,
                &execution,
                shape,
                alignment,
                eredu_core::InferenceGeometry {
                    batch_size: 1,
                    cached_positions: 0,
                    input_positions: 5,
                    max_output_tokens: 13,
                    prefill_chunk_positions: input.chunk.map_or(5, |n| n.get().min(5)),
                    output: eredu_core::OutputDemand::LastPosition,
                },
                &config,
            )?;
            let completed = strategy.prefill_cancellable(
                input, &mut cache,
                &mut eredu_architectures::speculative_execution::EmbeddedPredictionObservers::default(),
                &GenerationCancellationToken::new(), ReferenceEmbeddedContext { schedule: Some(&schedule), ..Default::default() },
            ).map_err(|error|error.to_string())?;
            assert!(matches!(
                completed,
                eredu_core::SpeculativePrefillOutcome::Complete(_)
            ));
            let trace = reference_trace();
            // These are actual operator invocations on materialized parameters,
            // not source inspection or a formula mirroring the implementation.
            let target_heads = trace
                .linear_outputs
                .iter()
                .filter(|(id, _)| matches!(id.as_str(), "lm_head.weight" | "head.weight"))
                .collect::<Vec<_>>();
            assert_eq!(
                target_heads.len(),
                1,
                "only the final selected target row projects scores"
            );
            assert_eq!(target_heads[0].1[1], 1);
            assert!(
                trace
                    .linear_outputs
                    .iter()
                    .all(|(id, _)| !id.ends_with("shared_head.head.weight")),
                "sequential cache seeding must not evaluate the prediction vocabulary head"
            );
        }
        let observers = match &self.captured {
            Some(case) => {
                eredu_architectures::speculative_execution::EmbeddedPredictionObservers::default()
                    .with_internal(ReferenceSpanObserver::new(case.clone()))
            }
            None => Default::default(),
        };
        let mut executor = eredu_architectures::speculative_execution::EmbeddedPredictionExecutor::<
            _,
            ReferenceEmbeddedMechanisms,
        >::with_observers(&mut strategy, observers);
        let mut executor = eredu_architectures::speculative_execution::DynEmbeddedExecutor::<
            ReferenceEmbeddedExecutorTypes,
        >::new(&mut executor);
        run_reference_embedded_scheduler(
            &self.selected,
            &execution,
            shape,
            alignment,
            &mut executor,
            self.captured.as_ref(),
        )
    }
}

fn run_reference_embedded_production(
    target_config: &serde_json::Value,
) -> Result<ReferenceProductionOutcome, String> {
    run_reference_embedded_production_with_snapshots(target_config, false)
}

fn run_reference_embedded_production_with_snapshots(
    target_config: &serde_json::Value,
    check_snapshot_context: bool,
) -> Result<ReferenceProductionOutcome, String> {
    run_reference_embedded_production_with_prefill(target_config, check_snapshot_context, None)
}

fn run_reference_embedded_production_with_prefill(
    target_config: &serde_json::Value,
    check_snapshot_context: bool,
    captured: Option<ReferenceCapturedCase>,
) -> Result<ReferenceProductionOutcome, String> {
    use std::num::NonZeroUsize;

    let artifact = reference_composite_artifact(target_config);
    let complete_inspection = eredu_architectures::configuration::inspect_artifact(artifact.path())
        .map_err(|error| error.to_string())?;
    let (target_plan, extension_plan) = complete_inspection
        .architecture_plan()
        .prediction_target_projection()
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "reference target did not retain its prediction extension".to_owned())?;
    let inspection = complete_inspection.map_architecture_plan(|_| target_plan);
    let capacity =
        eredu_architectures::prediction_extension::embedded_prediction_capacity(&extension_plan)
            .map_err(|error| error.to_string())?;
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|error| error.to_string())?,
        0,
    )
    .map_err(|error| error.to_string())?;
    let identity =
        |value| eredu_runtime::SpeculativeIdentity::new(value).map_err(|error| error.to_string());
    let contract = eredu_architectures::prediction_extension::embedded_speculative_contract(
        &extension_plan,
        eredu_architectures::prediction_extension::EmbeddedSpeculativeContractRequest::new(
            identity("reference-target")?,
            identity("reference-artifact")?,
            identity("safetensors")?,
            topology,
            identity("reference-text-processor")?,
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(
                captured
                    .as_ref()
                    .map_or(8, |case| case.prompt_positions.max(8)),
            )
            .unwrap(),
            capacity,
        ),
    )
    .map_err(|error| error.to_string())?;
    let mechanisms = eredu_runtime::SpeculativeMechanismCapabilities::new(
        contract
            .requirements()
            .mechanisms()
            .mechanisms()
            .iter()
            .copied(),
    );
    let store: eredu_checkpoint::store::RetainedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path())
            .map_err(|error| error.to_string())?,
    )
    .into();
    let requirements = eredu_architectures::routed_text::routed_text_requirements(&inspection)
        .map_err(|error| error.to_string())?;
    let selected_target = eredu_architectures::routed_text::select_routed_text_realization(
        &requirements,
        &eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
            eredu_runtime::ReplicatedTextSelectionRequest::new(
                eredu_runtime::LayerWeightResidency::FullyResident,
                eredu_runtime::CacheResidencyPolicy::Device,
            ),
            eredu_runtime::WeightResidency::fully_resident(),
        )
        .map_err(|error| error.to_string())?,
        &reference_text_capabilities(requirements.text()).with_exact_completion(true),
    )
    .map_err(|error| error.to_string())?;
    let prediction_tasks = selected_target
        .text()
        .auxiliary_materialization_tasks()
        .to_vec();
    let stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = stages.clone();
    let prepared = eredu_runtime::select_and_prepare_speculative_realization_observed(
        contract.requirements(),
        &contract.selection_request(eredu_runtime::SpeculativePlacementRequest::Single),
        &mechanisms,
        &move |stage| {
            observed.lock().unwrap().push(stage);
            Ok(())
        },
        |_| {
            eredu_architectures::prediction_extension::prepare_replicated_prediction_extension::<
                ReferenceBackend,
            >(&extension_plan, &prediction_tasks, &(), &())
            .map(|prepared| RefCell::new(Some(prepared)))
            .map_err(|error| error.to_string())
        },
        |_, prepared| {
            prepared
                .borrow_mut()
                .take()
                .ok_or_else(|| "reference prediction extension was materialized twice".to_owned())?
                .materialize::<ReferencePredictionMaterializer>(
                    &mut ReferencePredictionMaterializationContext {
                        store: store.as_ref(),
                    },
                )
                .map_err(|error| error.to_string())
        },
        |_, _| Ok::<_, String>(()),
        |_| Ok::<_, String>(()),
        |_, _| Ok::<_, String>(()),
    )
    .map_err(|error| format!("{error:?}"))?;
    let (selected, resources) = prepared.into_parts();
    let (_, extension, (), (), _) = resources.into_parts();
    let mut output = match extension_plan.kind() {
        eredu_architectures::configuration::PredictionExtensionKind::DeepSeekV3Mtp => {
            eredu_architectures::routed_text::visit_gated_routed_prediction_target_architecture::<
                ReferenceBackend,
                ReferenceState,
                ReferencePredictionMaterializer,
                _,
            >(
                &inspection,
                selected_target,
                extension,
                store,
                &(),
                RunReferenceEmbedded {
                    selected,
                    check_snapshot_context,
                    captured,
                },
            )
            .map_err(|error| error.to_string())?
        }
        eredu_architectures::configuration::PredictionExtensionKind::DeepSeekV4Embedded => {
            eredu_architectures::routed_text::visit_pooling_routed_prediction_target_architecture::<
                ReferenceBackend,
                ReferenceState,
                ReferencePredictionMaterializer,
                _,
            >(
                &inspection,
                selected_target,
                extension,
                store,
                &(),
                RunReferenceEmbedded {
                    selected,
                    check_snapshot_context,
                    captured,
                },
            )
            .map_err(|error| error.to_string())?
        }
        _ => return Err("reference embedded fixture selected an unexpected family".into()),
    };
    output.construction_stages = stages.lock().unwrap().clone();
    Ok(output)
}

#[allow(
    dead_code,
    reason = "owned by the unified reference_conformance target"
)]
pub(crate) fn embedded_snapshot_context_reaches_target_and_prediction_only_copies() {
    std::thread::Builder::new()
        .name("reference-embedded-snapshot-context".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            for config in [
                production_deepseek_v3_prediction_config(),
                production_deepseek_v4_dspark_config(),
            ] {
                let outcome = run_reference_embedded_production_with_snapshots(&config, true)
                    .expect("both snapshot branches must preserve their resource context");
                assert_construction_stages(&outcome);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

struct RunReferenceExternal {
    spans: Option<ReferenceExternalCase>,
    target: Box<dyn ReferenceExternalTarget>,
    cache: eredu_architectures::external_assistant::ExternalAssistantCache<ReferenceState>,
    capture: ExternalPredictionCaptureRequest,
    selected: eredu_runtime::SelectedSpeculativeRealization,
}

impl
    eredu_architectures::external_assistant::MaterializedExternalAssistantVisitor<
        ReferenceAssistantMaterializer,
    > for RunReferenceExternal
{
    type Output = Result<ReferenceProductionOutcome, String>;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        mut self,
        assistant: &mut ReferenceMaterializedAssistant<A>,
    ) -> Self::Output {
        if let Some(case) = &self.spans {
            assistant.observers =
                eredu_architectures::external_assistant::ExternalAssistantObservers::new(
                    ReferenceExternalSpanObserver {
                        case: case.clone(),
                        start: None,
                    },
                    eredu_runtime::NoopObserver,
                );
        }
        A::visit_executor::<ReferenceExternalMechanisms, _>(
            self.target.as_mut(),
            assistant,
            self.capture,
            RunReferenceScheduler {
                cache: &mut self.cache,
                spans: self.spans,
                selected: self.selected,
            },
        )
    }
}

trait ReferenceProductionSamplingContext: Clone {
    type Context<'a>: Copy;
}

impl ReferenceProductionSamplingContext for () {
    type Context<'a> = ();
}

#[derive(Clone, Default)]
struct ReferenceExternalSamplingContext;
impl ReferenceProductionSamplingContext for ReferenceExternalSamplingContext {
    type Context<'a> = ReferenceExternalContext;
}

#[derive(Clone, Default)]
struct ReferenceEmbeddedSamplingContext;

impl ReferenceProductionSamplingContext for ReferenceEmbeddedSamplingContext {
    type Context<'a> = ReferenceEmbeddedContext<'a>;
}

#[derive(Clone, Default)]
struct ReferenceProductionSampling<C = ()>(std::marker::PhantomData<C>);

impl<C: ReferenceProductionSamplingContext> SpeculativeSampling for ReferenceProductionSampling<C> {
    type Logits = u32;
    type Distribution = u32;
    type Seed = ();
    type RandomState = usize;
    type DraftRandomness = usize;
    type RandomnessRoot = usize;
    type Context<'a>
        = C::Context<'a>
    where
        Self: 'a;
    type Error = Error;

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn randomness_root<'a>(_: Option<()>, _: Self::Context<'a>) -> Result<usize, Error>
    where
        Self: 'a,
    {
        Ok(0)
    }

    fn target_randomness_from_root<'a>(
        root: &mut usize,
        _: Self::Context<'a>,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        let value = *root;
        *root += 1;
        Ok(value)
    }

    fn draft_randomness_from_root<'a>(
        root: &mut usize,
        _: Self::Context<'a>,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        let value = *root;
        *root += 1;
        Ok(value)
    }

    fn draft_randomness_at<'a>(
        root: &usize,
        position: SpeculativeDraftRandomPosition,
        _: Self::Context<'a>,
    ) -> Result<usize, Error>
    where
        Self: 'a,
    {
        Ok(*root + position.get())
    }

    fn process_logits<'a>(
        &mut self,
        logits: &u32,
        _: f32,
        _: &[u32],
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<u32, Error>
    where
        Self: 'a,
    {
        Ok(*logits)
    }

    fn sample<'a>(
        &self,
        distribution: &u32,
        _: f32,
        _: Option<&mut usize>,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<u32, Error>
    where
        Self: 'a,
    {
        Ok(*distribution)
    }

    fn probability_at<'a>(
        &self,
        distribution: &u32,
        token: u32,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<f32, Error>
    where
        Self: 'a,
    {
        Ok(if *distribution == token { 1.0 } else { 0.0 })
    }

    fn sample_unit_interval<'a>(
        &self,
        _: Option<&mut usize>,
        _: Self::Context<'a>,
    ) -> Result<f32, Error>
    where
        Self: 'a,
    {
        Ok(0.5)
    }

    fn positive_probability_difference<'a>(
        &self,
        target: &u32,
        _: &u32,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<Option<u32>, Error>
    where
        Self: 'a,
    {
        Ok(Some(*target))
    }

    fn update_sampler_state<'a>(
        &mut self,
        _: &u32,
        _: u32,
        _: SamplingPlacement,
        _: Self::Context<'a>,
    ) -> Result<(), Error>
    where
        Self: 'a,
    {
        Ok(())
    }
}

struct RunReferenceScheduler<'a> {
    spans: Option<ReferenceExternalCase>,
    cache: &'a mut eredu_architectures::external_assistant::ExternalAssistantCache<ReferenceState>,
    selected: eredu_runtime::SelectedSpeculativeRealization,
}

impl<'a, A>
    eredu_architectures::external_assistant::ExternalAssistantExecutorVisitor<
        A,
        ReferenceExternalMechanisms,
    > for RunReferenceScheduler<'a>
where
    A: eredu_architectures::ExternalAssistantArchitecture,
{
    type Output = Result<ReferenceProductionOutcome, String>;

    fn execute<'run, E>(self, executor: &'run mut E) -> Self::Output
    where
        Self: 'run,
        E: eredu_core::SpeculativeExecutor<
                Input = ReferencePreparedInput,
                Cache = eredu_architectures::external_assistant::ExternalAssistantCache<
                    ReferenceState,
                >,
                Logits = u32,
                Context<'run> = ReferenceExternalContext,
                Completion = ReferenceExternalCompletion,
                Telemetry = (),
                Error = Error,
            > + 'run,
    {
        let publications = Rc::new(Cell::new(0));
        let execution_stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed_execution = execution_stages.clone();
        let runtime = SpeculativeOutputRuntime::new(
            ReferenceProductionSampling::<ReferenceExternalSamplingContext>::default(),
            GenerationSequence::new(if self.spans.is_some() { 13 } else { 3 }, []),
            Constraint,
            Publisher {
                publications: publications.clone(),
            },
            self.spans
                .as_ref()
                .map(|case| case.cancellation.clone())
                .unwrap_or_default(),
        )
        .with_lifecycle_observer(std::sync::Arc::new(move |stage| {
            observed_execution.lock().unwrap().push(stage);
            record_reference_lifecycle(stage);
            Ok(())
        }));
        let config = SpeculativeConfig {
            max_tokens: if self.spans.is_some() { 13 } else { 3 },
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        };
        let lane = PreparedSpeculativeLane::new(
            self.cache,
            {
                let tokens = self.spans.as_ref().map(|case| {
                    [1, 3, 2, 4, 5]
                        .into_iter()
                        .cycle()
                        .take(case.prompt_positions)
                        .collect::<Vec<_>>()
                });
                let mut input = ReferencePreparedInput::tokens(tokens.as_deref().unwrap_or(&[9]))
                    .map_err(|e| e.to_string())?;
                input.chunk = self.spans.as_ref().and_then(|case| case.chunk);
                input.external_schedule = Some((self.selected, A::invocation_shape(), config.clone()));
                input
            },
            config,
            runtime,
            SpeculativeRandomness::new(None, None),
        );
        let completion_never = REFERENCE_COMPLETION_CONTROL.with(|slot| {
            slot.borrow()
                .as_ref()
                .is_some_and(|injection| matches!(injection.mode, ReferenceCompletionMode::Never))
        });
        let mut scheduler_options = SpeculativeSchedulerOptions::default().with_lookahead(false);
        if completion_never {
            scheduler_options = scheduler_options
                .with_completion_wait(
                    std::time::Duration::from_millis(1),
                    eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                )
                .map_err(|error| error.to_string())?;
        }
        let mut scheduler = SpeculativeScheduler::new(
            executor,
            scheduler_options,
            SpeculativeExecutionTopology::Single,
            false,
            false,
            ReferenceExternalContext::default(),
        )
        .map_err(|error| error.to_string())?;
        scheduler.submit(lane).map_err(|error| error.to_string())?;
        scheduler.run().map_err(|error| error.to_string())?;
        let mut completed = scheduler.finish().map_err(|error| error.to_string())?;
        let request = completed
            .take_requests()
            .pop()
            .ok_or_else(|| "reference scheduler returned no request".to_owned())?;
        let execution_stages = execution_stages.lock().unwrap().clone();
        Ok(ReferenceProductionOutcome {
            tokens: request.token_ids().to_vec(),
            accepted: request.stats().accept_lens().to_vec(),
            target_tokens: request.stats().target_tokens(),
            publications: publications.get(),
            construction_stages: Vec::new(),
            execution_stages,
        })
    }
}

fn run_reference_external_production(
    target_config: &serde_json::Value,
    assistant_artifact: &std::path::Path,
) -> Result<ReferenceProductionOutcome, String> {
    run_reference_external_spans(target_config, assistant_artifact, None)
}

fn run_reference_external_spans(
    target_config: &serde_json::Value,
    assistant_artifact: &std::path::Path,
    spans: Option<ReferenceExternalCase>,
) -> Result<ReferenceProductionOutcome, String> {
    use std::num::NonZeroUsize;

    let (mut target, _target_artifact) = construct_reference_composite_target(target_config);
    let compatible = eredu_architectures::prepare_external_assistant(assistant_artifact)
        .map_err(|error| error.to_string())?
        .select_materialization(
            None,
            eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS,
            |_, _| Some(eredu_runtime::WeightLoweringKind::Direct),
        )?
        .prove_target_compatibility(&target.profile())?;
    let capture = compatible.capture().clone();
    let fingerprint = [33_u8; 32];
    let tokenizer = eredu_core::TokenizerCompatibilityProof::prove(fingerprint, fingerprint)
        .map_err(|error| error.to_string())?;
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).map_err(|error| error.to_string())?,
        0,
    )
    .map_err(|error| error.to_string())?;
    let contract = compatible
        .speculative_contract(
            eredu_architectures::external_assistant::ExternalSpeculativeContractRequest::new(
                topology,
                eredu_runtime::SpeculativeIdentity::new("reference-processor-v1")
                    .map_err(|error| error.to_string())?,
                tokenizer,
                fingerprint,
                NonZeroUsize::new(2).unwrap(),
            ),
        )
        .map_err(|error| error.to_string())?;
    let capabilities = eredu_runtime::SpeculativeMechanismCapabilities::new(
        contract
            .requirements()
            .mechanisms()
            .mechanisms()
            .iter()
            .copied(),
    );
    let compatible = compatible
        .prepare_source(eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS)
        .map_err(|error| error.to_string())?;
    let payload = RefCell::new(Some(compatible));
    let construction_stages = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_stages = construction_stages.clone();
    let observer = move |stage| {
        observed_stages.lock().unwrap().push(stage);
        Ok(())
    };
    let prepared = eredu_runtime::select_and_prepare_speculative_realization_observed(
        contract.requirements(),
        &contract.selection_request(eredu_runtime::SpeculativePlacementRequest::Single),
        &capabilities,
        &observer,
        |_| Ok::<_, String>(&payload),
        |_, payload| {
            payload
                .borrow_mut()
                .take()
                .ok_or_else(|| "reference assistant payload was opened twice".to_owned())?
                .visit(ReferenceAssistantMaterializer)
                .map_err(|error| error.to_string())
        },
        |_, _| target.prepare_cache().map_err(|error| error.to_string()),
        |_| Ok::<_, String>(()),
        |_, _| Ok::<_, String>(()),
    )
    .map_err(|error| error.to_string())?;
    let (selected, resources) = prepared.into_parts();
    let (_, mut assistant, native, (), transfer) = resources.into_parts();
    assert!(transfer.is_none());
    let cache =
        eredu_architectures::external_assistant::ExternalAssistantCache::new(native, selected.clone());
    let mut outcome = assistant.visit(RunReferenceExternal {
        target,
        spans,
        cache,
        capture,
        selected,
    })?;
    outcome.construction_stages = construction_stages.lock().unwrap().clone();
    Ok(outcome)
}

include!("reference_captured_prefill.rs");

include!("reference_external_prefill.rs");
