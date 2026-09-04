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

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: eredu_architectures::prediction_extension::PreparedPredictionUnit<M>,
        _: Option<&eredu_runtime::LocalModelLayout>,
    ) -> Result<Self::Module<M>, Self::Error>
    where
        M: Parameterized<ReferenceTensor>,
    {
        let (_, mut local, mut recipes) = prepared.into_parts();
        struct Bindings<'a> {
            values: Vec<eredu_runtime::WeightBinding>,
            recipes: &'a mut std::collections::BTreeMap<
                String,
                eredu_checkpoint::recipe::DerivedWeightRecipe,
            >,
        }
        impl<'a, 'r> ParameterVisitor<'a, ReferenceTensor> for Bindings<'r> {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a ReferenceTensor) {
                let bytes = value
                    .shape()
                    .iter()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .product::<u64>()
                    * 4;
                let binding = match self.recipes.remove(metadata.id.as_str()) {
                    Some(recipe) => eredu_runtime::WeightBinding::from_recipe(
                        metadata.id.as_str(),
                        recipe,
                        bytes,
                    ),
                    None => eredu_runtime::WeightBinding::new(
                        metadata.id.as_str(),
                        metadata.id.as_str(),
                        eredu_checkpoint::store::TensorSelection::Full,
                        bytes,
                    ),
                };
                self.values.push(binding.unwrap());
            }
        }
        let mut bindings = Bindings {
            values: Vec::new(),
            recipes: &mut recipes,
        };
        local.visit_parameters(&mut bindings);
        let materialized = eredu_runtime::materialize_bindings::<ReferenceBackend>(
            context.store,
            &bindings.values,
            &(),
        )
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
}

struct ReferenceReplicatedMechanisms;

impl<A> ReplicatedTextSessionMechanisms<A, ReferenceBackend> for ReferenceReplicatedMechanisms
where
    A: eredu_runtime::LayeredArchitecture<ReferenceBackend, ReferenceState, Error = Error>,
{
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

    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        _: &(),
    ) -> Result<Self::StateCheckpoint, Self::Error> {
        Ok(state.clone())
    }

    fn restore_state(
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
        _: &ReferenceTensor,
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
        Ok(Self { input, identity })
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
    type Context<'a> = ();
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

    fn executor_context<'a>(
        _: <Self::ExecutorTypes as eredu_architectures::speculative_execution::EmbeddedExecutorTypes>::Context<'a>,
    ) -> <ReferenceEmbeddedMechanisms as eredu_architectures::speculative_execution::SpeculativeTensorMechanisms>::Context<'a>{
    }

    fn target_context<'a>(_: ()) -> &'a () {
        &()
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

    fn token(_: u32, _: &()) -> Result<ReferenceTensor, Error> {
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

    fn take_telemetry() -> Result<Self::Telemetry, Error> {
        Ok(())
    }
}

impl eredu_architectures::speculative_execution::SpeculativeTensorMechanisms
    for ReferenceEmbeddedMechanisms
{
    type Tensor = ReferenceTensor;
    type Logits = u32;
    type Context<'a> = ();
    type Completion = ReferenceExternalCompletion;
    type Error = Error;

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
        _: &eredu_architectures::speculative_execution::EmbeddedPredictionOutput<Self::Tensor>,
        _: &Self::Tensor,
        _: Self::Context<'a>,
    ) -> Result<Self::Completion, Self::Error> {
        Ok(ReferenceExternalCompletion)
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
        _: eredu_checkpoint::store::SharedCheckpointSource,
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
        _: eredu_checkpoint::store::SharedCheckpointSource,
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
    let capabilities = reference_text_capabilities(requirements.execution());
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
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path()).unwrap(),
    );
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
        prepared: eredu_architectures::PreparedExternalAssistant<A>,
    ) -> Result<Self::Output<A>, Self::Error> {
        let (checkpoint, config) = prepared.into_parts();
        let store: eredu_checkpoint::store::SharedCheckpointSource = match checkpoint {
            eredu_architectures::ExternalAssistantCheckpoint::SafeTensors {
                source,
                catalog,
                resolution,
                ..
            } => {
                let store = eredu_checkpoint::store::SafetensorsWeightStore::open(&source)
                    .map_err(|error| Error::backend(error.to_string()))?;
                if resolution.source_keys().is_empty()
                    || catalog.len() < resolution.source_keys().len()
                {
                    return Err(Error::backend(
                        "reference assistant payload did not preserve its admitted resolution",
                    ));
                }
                std::sync::Arc::new(store)
            }
            eredu_architectures::ExternalAssistantCheckpoint::Gguf { .. } => {
                return Err(Error::backend(
                    "reference production proof requires its SafeTensors fixture",
                ));
            }
        };
        let mut module = A::module::<ReferenceBackend>(config.clone(), &())?;
        struct Bindings(Vec<eredu_runtime::WeightBinding>);
        impl<'a> ParameterVisitor<'a, ReferenceTensor> for Bindings {
            fn visit(&mut self, metadata: ParameterMetadata, value: &'a ReferenceTensor) {
                let expected_bytes = value
                    .shape()
                    .iter()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .product::<u64>()
                    * 4;
                self.0.push(
                    eredu_runtime::WeightBinding::new(
                        metadata.id.as_str(),
                        metadata.id.as_str(),
                        eredu_checkpoint::store::TensorSelection::Full,
                        expected_bytes,
                    )
                    .unwrap(),
                );
            }
        }
        let mut bindings = Bindings(Vec::new());
        module.visit_parameters(&mut bindings);
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

#[derive(Debug)]
struct ReferenceExternalCompletion;

impl eredu_core::Completion for ReferenceExternalCompletion {
    type Error = Error;

    fn is_complete(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), Self::Error> {
        Ok(())
    }
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
    type Context<'a> = ();
    type Completion = ReferenceExternalCompletion;
    type Telemetry = ();
    type Error = Error;

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
        _: impl IntoIterator<Item = &'a Self::Tensor>,
    ) -> Result<Self::Completion, Self::Error>
    where
        Self::Tensor: 'a,
    {
        Ok(ReferenceExternalCompletion)
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
}

fn run_reference_embedded_scheduler(
    selected: &eredu_runtime::SelectedSpeculativeRealization,
    executor: &mut eredu_architectures::speculative_execution::DynEmbeddedExecutor<
        '_,
        ReferenceEmbeddedExecutorTypes,
    >,
) -> Result<ReferenceProductionOutcome, String> {
    let mut cache = executor.new_cache().map_err(|error| error.to_string())?;
    let publications = Rc::new(Cell::new(0));
    let runtime = SpeculativeOutputRuntime::new(
        ReferenceProductionSampling,
        GenerationSequence::new(3, []),
        Constraint,
        Publisher {
            publications: publications.clone(),
        },
        GenerationCancellationToken::new(),
    );
    let lane = PreparedSpeculativeLane::new(
        &mut cache,
        ReferencePreparedInput::tokens(&[9]).map_err(|error| error.to_string())?,
        SpeculativeConfig {
            max_tokens: 3,
            max_draft_tokens: selected.requirements().strategy().proposal_capacity().get(),
            temperature: 0.0,
            eos_token_ids: Vec::new(),
        },
        runtime,
        SpeculativeRandomness::new(None, None),
    );
    let mut scheduler = SpeculativeScheduler::new(
        executor,
        SpeculativeSchedulerOptions::default().with_lookahead(false),
        SpeculativeExecutionTopology::Single,
        false,
        false,
        (),
    )
    .map_err(|error| format!("{error:?}"))?;
    scheduler.submit(lane).map_err(|error| error.to_string())?;
    scheduler.run().map_err(|error| error.to_string())?;
    let mut completed = scheduler.finish().map_err(|error| error.to_string())?;
    let request = completed
        .take_requests()
        .pop()
        .ok_or_else(|| "reference embedded scheduler returned no request".to_owned())?;
    Ok(ReferenceProductionOutcome {
        tokens: request.token_ids().to_vec(),
        accepted: request.stats().accept_lens().to_vec(),
        target_tokens: request.stats().target_tokens(),
        publications: publications.get(),
        construction_stages: Vec::new(),
    })
}

struct RunReferenceEmbedded {
    selected: eredu_runtime::SelectedSpeculativeRealization,
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
        _: eredu_checkpoint::store::SharedCheckpointSource,
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
        let mut executor = eredu_architectures::speculative_execution::EmbeddedPredictionExecutor::<
            _,
            ReferenceEmbeddedMechanisms,
        >::new(&mut strategy);
        let mut executor = eredu_architectures::speculative_execution::DynEmbeddedExecutor::<
            ReferenceEmbeddedExecutorTypes,
        >::new(&mut executor);
        run_reference_embedded_scheduler(&self.selected, &mut executor)
    }
}

fn run_reference_embedded_production(
    target_config: &serde_json::Value,
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
            NonZeroUsize::new(8).unwrap(),
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
    let store: eredu_checkpoint::store::SharedCheckpointSource = std::sync::Arc::new(
        eredu_checkpoint::store::SafetensorsWeightStore::open(artifact.path())
            .map_err(|error| error.to_string())?,
    );
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
            >(&extension_plan, store.as_ref(), &(), &())
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
        &reference_text_capabilities(requirements.text()),
    )
    .map_err(|error| error.to_string())?;
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
                RunReferenceEmbedded { selected },
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
                RunReferenceEmbedded { selected },
            )
            .map_err(|error| error.to_string())?
        }
        _ => return Err("reference embedded fixture selected an unexpected family".into()),
    };
    output.construction_stages = stages.lock().unwrap().clone();
    Ok(output)
}

struct RunReferenceExternal {
    target: Box<dyn ReferenceExternalTarget>,
    cache: eredu_architectures::external_assistant::ExternalAssistantCache<ReferenceState>,
    capture: ExternalPredictionCaptureRequest,
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
        A::visit_executor::<ReferenceExternalMechanisms, _>(
            self.target.as_mut(),
            assistant,
            self.capture,
            RunReferenceScheduler {
                cache: &mut self.cache,
            },
        )
    }
}

#[derive(Clone, Default)]
struct ReferenceProductionSampling;

impl SpeculativeSampling for ReferenceProductionSampling {
    type Logits = u32;
    type Distribution = u32;
    type Seed = ();
    type RandomState = usize;
    type DraftRandomness = usize;
    type RandomnessRoot = usize;
    type Context<'a> = ();
    type Error = Error;

    fn supports_exact_optimistic_promotion(&self) -> bool {
        true
    }

    fn randomness_root<'a>(_: Option<()>, _: ()) -> Result<usize, Error>
    where
        Self: 'a,
    {
        Ok(0)
    }

    fn target_randomness_from_root<'a>(root: &mut usize, _: ()) -> Result<usize, Error>
    where
        Self: 'a,
    {
        let value = *root;
        *root += 1;
        Ok(value)
    }

    fn draft_randomness_from_root<'a>(root: &mut usize, _: ()) -> Result<usize, Error>
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
        _: (),
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
        _: (),
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
        _: (),
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
        _: (),
    ) -> Result<f32, Error>
    where
        Self: 'a,
    {
        Ok(if *distribution == token { 1.0 } else { 0.0 })
    }

    fn sample_unit_interval<'a>(&self, _: Option<&mut usize>, _: ()) -> Result<f32, Error>
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
        _: (),
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
        _: (),
    ) -> Result<(), Error>
    where
        Self: 'a,
    {
        Ok(())
    }
}

struct RunReferenceScheduler<'a> {
    cache: &'a mut eredu_architectures::external_assistant::ExternalAssistantCache<ReferenceState>,
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
                Context<'run> = (),
                Completion = ReferenceExternalCompletion,
                Telemetry = (),
                Error = Error,
            > + 'run,
    {
        let publications = Rc::new(Cell::new(0));
        let runtime = SpeculativeOutputRuntime::new(
            ReferenceProductionSampling,
            GenerationSequence::new(3, []),
            Constraint,
            Publisher {
                publications: publications.clone(),
            },
            GenerationCancellationToken::new(),
        );
        let lane = PreparedSpeculativeLane::new(
            self.cache,
            ReferencePreparedInput::tokens(&[9]).map_err(|error| error.to_string())?,
            SpeculativeConfig {
                max_tokens: 3,
                max_draft_tokens: 2,
                temperature: 0.0,
                eos_token_ids: Vec::new(),
            },
            runtime,
            SpeculativeRandomness::new(None, None),
        );
        let mut scheduler = SpeculativeScheduler::new(
            executor,
            SpeculativeSchedulerOptions::default().with_lookahead(false),
            SpeculativeExecutionTopology::Single,
            false,
            false,
            (),
        )
        .map_err(|error| error.to_string())?;
        scheduler.submit(lane).map_err(|error| error.to_string())?;
        scheduler.run().map_err(|error| error.to_string())?;
        let mut completed = scheduler.finish().map_err(|error| error.to_string())?;
        let request = completed
            .take_requests()
            .pop()
            .ok_or_else(|| "reference scheduler returned no request".to_owned())?;
        Ok(ReferenceProductionOutcome {
            tokens: request.token_ids().to_vec(),
            accepted: request.stats().accept_lens().to_vec(),
            target_tokens: request.stats().target_tokens(),
            publications: publications.get(),
            construction_stages: Vec::new(),
        })
    }
}

fn run_reference_external_production(
    target_config: &serde_json::Value,
    assistant_artifact: &std::path::Path,
) -> Result<ReferenceProductionOutcome, String> {
    use std::num::NonZeroUsize;

    let (mut target, _target_artifact) = construct_reference_composite_target(target_config);
    let compatible = eredu_architectures::prepare_external_assistant(assistant_artifact)
        .map_err(|error| error.to_string())?
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
        eredu_architectures::external_assistant::ExternalAssistantCache::new(native, selected);
    let mut outcome = assistant.visit(RunReferenceExternal {
        target,
        cache,
        capture,
    })?;
    outcome.construction_stages = construction_stages.lock().unwrap().clone();
    Ok(outcome)
}
