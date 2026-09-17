//! Real scalar embedded prediction through retained preparation and the common driver.

use super::*;
use eredu_architectures::{
    prediction_extension::*,
    prepared_execution::*,
    replicated_text::{ReplicatedPredictionProfileDispatcher, ReplicatedPredictionTargetVisitor},
    PreparationMechanismProvider,
};
use std::rc::Rc;

#[path = "prediction_adapter/inkling_components.rs"]
mod inkling_components;
#[path = "prediction_adapter/invocation.rs"]
mod invocation;
#[path = "prediction_adapter/qwen_components.rs"]
mod qwen_components;
#[path = "prediction_adapter/residency.rs"]
mod residency;
#[path = "prediction_adapter/v3_components.rs"]
mod v3_components;

#[path = "prediction_adapter/retained_resources.rs"]
mod retained_resources;
#[path = "prediction_adapter/startup.rs"]
mod startup;

type State = DeviceState<NumericBackend, NumericHybridLayerState>;
type Snapshots = Rc<RefCell<Vec<State>>>;

struct Module<M>(M);
impl<M> AsMut<M> for Module<M> {
    fn as_mut(&mut self) -> &mut M {
        &mut self.0
    }
}

struct PredictionState {
    state: State,
    snapshots: Snapshots,
}
impl Clone for PredictionState {
    fn clone(&self) -> Self {
        // Capture the actual retained scalar tensors whenever the generic lane
        // owner checkpoints it; no family state geometry is reconstructed.
        self.snapshots.borrow_mut().push(self.state.clone());
        Self {
            state: self.state.clone(),
            snapshots: Rc::clone(&self.snapshots),
        }
    }
}
impl RuntimeState<NumericBackend> for PredictionState {
    type RetainedValues<'a> = <State as RuntimeState<NumericBackend>>::RetainedValues<'a>;
    fn layout(&self) -> &eredu_runtime::StateLayout {
        self.state.layout()
    }
    fn visit_all_retained_values(
        &self,
        visitor: &mut dyn FnMut(&NumericTensor),
    ) -> Result<(), StateError> {
        self.state.visit_all_retained_values(visitor)
    }
    fn retained_values(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        self.state.retained_values(ordinal, address)
    }
}
impl PredictionModelState<NumericBackend> for PredictionState {
    type LayerState = NumericHybridLayerState;
    fn prediction_layers_mut(&mut self) -> &mut [Self::LayerState] {
        self.state.as_mut()
    }
}

struct Materializer;
struct Materialization<'a> {
    source: RetainedCheckpointSource,
    context: &'a NumericContext,
    snapshots: Snapshots,
}
impl PredictionExtensionMaterializer<NumericBackend> for Materializer {
    type Error = String;
    type Module<M> = Module<M>;
    type PoolingState = NumericPoolingCache;
    type SequentialState = NumericCompressedCache;
    type ModelState = PredictionState;
    type Context<'a> = Materialization<'a>;
    type SnapshotContext<'a> = &'a NumericContext;
    fn complete_prediction_values<'a>(
        values: impl IntoIterator<Item = &'a NumericTensor>,
        _context: &<NumericTensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), eredu_core::BackendFailure> {
        invocation::complete(values)
    }

    fn materialize_module<M>(
        context: &mut Self::Context<'_>,
        prepared: PreparedPredictionUnit<M>,
        _layout: Option<&LocalModelLayout>,
    ) -> Result<Self::Module<M>, String>
    where
        M: Parameterized<NumericTensor>,
    {
        if context.context.partition.is_some() {
            return Err("scalar embedded proof has no partitioned module lowering".into());
        }
        let (_, mut module, tasks) = prepared.into_parts();
        let bound = payload::bind_module(
            &mut module,
            &tasks,
            context.source.as_ref(),
            context.context,
        )
        .map_err(|error| error.to_string())?;
        assert!(bound > 0);
        record_reference_materialization_tasks(tasks.len());
        Ok(Module(module))
    }
    fn pooling_state(
        _: &mut Self::Context<'_>,
        _: usize,
        _: LayerCachePolicy,
    ) -> Result<Self::PoolingState, String> {
        Err("this scalar prediction profile does not admit pooling state".into())
    }
    fn model_state(
        context: &mut Self::Context<'_>,
        layout: eredu_runtime::StateLayout,
    ) -> Result<Self::ModelState, String> {
        let state = State::create(layout, |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .map_err(|error| error.to_string())?;
        Ok(PredictionState {
            state,
            snapshots: Rc::clone(&context.snapshots),
        })
    }
    fn sequential_state() -> Self::SequentialState {
        NumericCompressedCache::resident()
    }
}

struct Provider;
impl PreparationMechanismProvider for Provider {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        prepared_adapter::NumericPreparationProvider { addressable: false }
            .preparation_capabilities()
    }
    fn supports_grouped_operation(
        &self,
        operation: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        prepared_adapter::NumericPreparationProvider { addressable: false }
            .supports_grouped_operation(operation)
    }
    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        prepared_adapter::NumericPreparationProvider { addressable: false }
            .replicated_text_capabilities(requirements, request)
    }
    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        prepared_adapter::NumericPreparationProvider { addressable: false }.processor_capabilities()
    }
    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        use eredu_runtime::SpeculativeMechanism::*;
        eredu_runtime::SpeculativeMechanismCapabilities::new([
            TensorOperations,
            NeuralOperations,
            GroupedNeuralOperations,
            PayloadMaterialization,
            LogitsProcessing,
            Sampling,
            Randomness,
            StateStorage,
            StorageResidency,
            ExactCompletion,
            Observation,
            QueueBinding,
            Communication,
            Agreement,
            Publication,
        ])
    }
    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        numeric_partition_capabilities()
    }
}

type ResidentPolicy<A> =
    <NumericReplicatedMechanisms as eredu_runtime::ReplicatedTextSessionMechanisms<
        A,
        NumericBackend,
    >>::ResidentPolicy;
type BoundedPolicy<A> =
    <NumericReplicatedMechanisms as eredu_runtime::ReplicatedTextSessionMechanisms<
        A,
        NumericBackend,
    >>::BoundedPolicy;
trait Driver<A>:
    eredu_runtime::ReplicatedTextExecutionStrategy<
    A,
    NumericBackend,
    State,
    ResidentPolicy<A>,
    BoundedPolicy<A>,
>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
{
}
impl<A, D> Driver<A> for D
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
    D: eredu_runtime::ReplicatedTextExecutionStrategy<
        A,
        NumericBackend,
        State,
        ResidentPolicy<A>,
        BoundedPolicy<A>,
    >,
{
}

struct Invoker<'a, A, D = eredu_runtime::DirectReplicatedTextExecution>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
    D: Driver<A>,
{
    session: &'a mut eredu_runtime::ReplicatedTextSession<
        A,
        NumericBackend,
        NumericReplicatedMechanisms,
        D,
    >,
    context: &'a NumericContext,
    fail_after: bool,
}
struct FailAfter<O>(O);
impl<A, O> eredu_runtime::PredictionTargetOperation<A, NumericBackend, State> for FailAfter<O>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    O: eredu_runtime::PredictionTargetOperation<A, NumericBackend, State>,
{
    type Output = O::Output;
    fn preserves_architecture_declarations(&self) -> bool {
        self.0.preserves_architecture_declarations()
    }
    fn apply(
        self,
        architecture: &mut A,
        state: &mut State,
        parallel: Option<&NumericParallelContext>,
        context: &NumericContext,
    ) -> Result<Self::Output, Error> {
        let _ = self.0.apply(architecture, state, parallel, context)?;
        // The failure is after real extension execution, not a preflight guard.
        Err(Error::backend(
            "injected failure after prediction execution",
        ))
    }
}
impl<A, D> PredictionOperationInvoker<A, NumericBackend, State> for Invoker<'_, A, D>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
    D: Driver<A>,
{
    type Error = String;
    fn invoke<O>(&mut self, operation: O) -> Result<O::Output, String>
    where
        O: eredu_runtime::PredictionTargetOperation<A, NumericBackend, State>,
    {
        if self.fail_after {
            self.session
                .apply_prediction_target_operation(FailAfter(operation), self.context)
                .map_err(|error| error.to_string())
        } else {
            self.session
                .apply_prediction_target_operation(operation, self.context)
                .map_err(|error| error.to_string())
        }
    }
    fn invalid(message: String) -> String {
        message
    }
}

struct Run {
    target: NumericTensor,
    capture: NumericTensor,
    proposal: NumericTensor,
    proposal_capture: NumericTensor,
    target_state: State,
    lane_state: State,
}
struct Visitor<'a> {
    context: &'a NumericContext,
    binding: PredictionBinding,
    snapshots: Snapshots,
    target_keys: BTreeSet<String>,
    component_scopes: Vec<eredu_core::component::ComponentExecutionScope>,
}
impl ReplicatedPredictionProfileDispatcher<NumericBackend, Materializer> for Visitor<'_> {
    type Output = Run;
    type Error = String;
    type State = State;
    type Visitor = Self;
    fn into_visitor(self) -> Self {
        self
    }
}
impl ReplicatedPredictionTargetVisitor<NumericBackend, State, Materializer> for Visitor<'_> {
    type Output = Run;
    type Error = String;
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        extension: <A as MaterializedPredictionTarget<NumericBackend>>::Extension<Materializer>,
        checkpoint: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + MaterializedPredictionTarget<NumericBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        assert_eq!(
            checkpoint
                .source_keys()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            self.target_keys
        );
        let (session, _) = construct_selected_text_session::<NumericBackend, _, _>(
            prepared,
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint),
            self.context,
        )?;
        self.run_session(session, extension)
    }
}

impl Visitor<'_> {
    fn run_session<A, D, E>(
        self,
        mut session: eredu_runtime::ReplicatedTextSession<
            A,
            NumericBackend,
            NumericReplicatedMechanisms,
            D,
        >,
        mut extension: E,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>,
        A::StaticModules: Clone,
        D: Driver<A>,
        E: MaterializedPredictionExecutor<A, NumericBackend, Materializer>,
    {
        let snapshots_before = self.snapshots.borrow().len();
        retained_resources::verify::<A, E>(&mut extension, 1);
        assert_eq!(
            self.snapshots.borrow().len(),
            snapshots_before,
            "inspection never clones a state prototype"
        );
        let tokens = NumericTensor::token_ids(&[1, 3, 2]);
        let (target, capture) = session
            .prefill_prediction_target(&tokens, None, self.context)
            .map_err(|error| error.to_string())?;
        let lane_id = self.binding.selected().lane_identity(
            eredu_runtime::SpeculativeIdentity::new("numeric-prompt-1-3-2").unwrap(),
            7,
        );
        extension
            .validate_capture(self.binding.selected(), &lane_id, &capture.shape)
            .map_err(|error| error.to_string())?;
        let mut bad_shape = capture.shape.clone();
        *bad_shape.last_mut().unwrap() += 1;
        assert!(extension
            .validate_capture(self.binding.selected(), &lane_id, &bad_shape)
            .is_err());
        assert!(extension.depth() > 0);
        let mut lane = extension.new_state();
        let mut invoker = Invoker {
            session: &mut session,
            context: self.context,
            fail_after: false,
        };
        extension.prefill::<State, _>(&mut invoker, &capture, &capture, &tokens, &mut lane)?;
        if extension.supports_internal_observations() {
            let _ = lane.clone();
            let ordinary_state = self.snapshots.borrow().last().unwrap().clone();
            let mut observed_lane = extension.new_state();
            let mut observer = qwen_components::Observe::default();
            extension.prefill_observed::<State, _>(
                &mut invoker,
                &capture,
                &capture,
                &tokens,
                &mut observed_lane,
                Some(&mut observer),
            )?;
            let _ = observed_lane.clone();
            assert_state_exact(
                self.snapshots.borrow().last().unwrap(),
                &ordinary_state,
                ordinary_state.layout().len(),
                "prepared observed prefill retains exact cache",
            );
            assert_eq!(
                observer.values[&self.component_scopes[0].components[0].activation].shape[1],
                3
            );
            assert_eq!(
                observer.values[&self.component_scopes[0].readout.linear_scores].shape[1],
                3
            );
        }

        let checkpoint = lane.clone();
        let prior = capture.axis_slice(1, 2, 3);
        let token = NumericTensor::token_ids(&[4]);
        let (proposal, proposal_capture) =
            extension.logits::<State, _>(&mut invoker, &prior, &token, 0, &mut lane)?;
        let _ = lane.clone();
        let expected_lane = self.snapshots.borrow().last().unwrap().clone();
        if extension.supports_internal_observations() {
            qwen_components::verify(
                &self.component_scopes,
                &mut extension,
                &mut invoker,
                &checkpoint,
                &prior,
                &token,
                &proposal,
                &proposal_capture,
            )?;
        }

        lane = checkpoint.clone();
        let target_before = invoker
            .session
            .checkpoint(self.context)
            .map_err(|error| error.to_string())?;
        invoker.fail_after = true;
        assert!(extension
            .logits::<State, _>(&mut invoker, &prior, &token, 0, &mut lane)
            .unwrap_err()
            .contains("injected failure after prediction execution"));
        let _ = lane.clone();
        assert_state_exact(
            self.snapshots.borrow().last().unwrap(),
            &expected_lane,
            expected_lane.layout().len(),
            "injected failure follows real lane mutation",
        );
        let target_after = invoker
            .session
            .report()
            .map_err(|error| error.to_string())?
            .state_report()
            .clone();
        assert_state_exact(
            &target_after,
            &target_before,
            target_before.layout().len(),
            "failed prediction restores target state",
        );
        // Lane storage is separate from canonical target state; its owner
        // restores the exact cloned checkpoint before retrying a failed proposal.
        lane = checkpoint;
        invoker.fail_after = false;
        let (retried, retried_capture) =
            extension.logits::<State, _>(&mut invoker, &prior, &token, 0, &mut lane)?;
        assert_tensor_exact(&retried, &proposal, "prediction checkpoint retry logits");
        assert_tensor_exact(
            &retried_capture,
            &proposal_capture,
            "prediction checkpoint retry capture",
        );
        let _ = lane.clone();
        let lane_state = self.snapshots.borrow().last().unwrap().clone();
        assert_state_exact(
            &lane_state,
            &expected_lane,
            lane_state.layout().len(),
            "prediction checkpoint retry retained tensors",
        );
        let target_state = invoker
            .session
            .report()
            .map_err(|error| error.to_string())?
            .state_report()
            .clone();
        assert_state_exact(
            &target_state,
            &target_before,
            target_state.layout().len(),
            "prediction never advances canonical target cache",
        );
        let physical_per_depth = lane_state.layout().len() / extension.depth();
        assert_eq!(
            physical_per_depth * extension.depth(),
            lane_state.layout().len()
        );
        for (physical, layer) in lane_state.as_ref().iter().enumerate() {
            let expected = if matches!(
                lane_state.layout().layer(physical),
                Some(LayerCachePolicy::NoState)
            ) {
                0
            } else if physical < physical_per_depth {
                4
            } else {
                3
            };
            assert_eq!(
                layer.position(),
                expected,
                "physical prediction state {physical}"
            );
        }
        assert!(target_state
            .as_ref()
            .iter()
            .enumerate()
            .all(|(physical, layer)| layer.position()
                == if matches!(
                    target_state.layout().layer(physical),
                    Some(LayerCachePolicy::NoState)
                ) {
                    0
                } else {
                    3
                }));
        Ok(Run {
            target,
            capture,
            proposal,
            proposal_capture,
            target_state,
            lane_state,
        })
    }
}

impl
    eredu_architectures::routed_text::RoutedPredictionProfileDispatcher<
        NumericBackend,
        Materializer,
    > for Visitor<'_>
{
    type Output = Run;
    type Error = String;
    type GatedState = State;
    type PoolingState = State;
    type GatedVisitor = Self;
    type PoolingVisitor = Self;
    fn into_gated_visitor(self) -> Self {
        self
    }
    fn into_pooling_visitor(self) -> Self {
        self
    }
}
impl
    eredu_architectures::routed_text::RoutedPredictionTargetVisitor<
        NumericBackend,
        State,
        Materializer,
    > for Visitor<'_>
{
    type Output = Run;
    type Error = String;
    fn visit<A>(
        self,
        prepared: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        extension: <A as MaterializedPredictionTarget<NumericBackend>>::Extension<Materializer>,
        checkpoint: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + MaterializedPredictionTarget<NumericBackend>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        assert_eq!(
            checkpoint
                .source_keys()
                .into_iter()
                .collect::<BTreeSet<_>>(),
            self.target_keys
        );
        let session = prepared
            .construct_resident_session::<NumericBackend, NumericReplicatedMechanisms>(
                NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint),
                self.context,
            )?;
        self.run_session(session, extension)
    }
}

struct Ordinary;
impl ReplicatedTextArchitectureVisitor<NumericBackend, State> for Ordinary {
    type Output = Run;
    type Error = String;
    fn construction_started(&mut self) {}
    fn visit<A>(
        self,
        _: PreparedReplicatedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        Err("prediction selection unexpectedly reached ordinary construction".into())
    }
}
impl eredu_architectures::routed_text::RoutedTextArchitectureVisitor<NumericBackend, State>
    for Ordinary
{
    type Output = Run;
    type Error = String;
    fn visit<A>(
        self,
        _: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        Err("prediction selection reached ordinary routed construction".into())
    }
}
impl eredu_architectures::Relu2RoutedTextArchitectureVisitor<NumericBackend, State> for Ordinary {
    type Output = Run;
    type Error = String;
    fn visit<A>(
        self,
        _: eredu_architectures::routed_text::PreparedRoutedTextArchitecture<A>,
        _: RetainedCheckpointSource,
    ) -> Result<Run, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + eredu_runtime::RoutedLayeredArchitecture<NumericBackend, State>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        Err("prediction selection reached ordinary ReLU routed construction".into())
    }
}

struct Assembler;
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Run;
    type Output = Run;
    type Error = String;
    fn floating_state_dtype(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, String> {
        Ok(eredu_runtime::StateStorageDtype::F32)
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("scalar embedded route has no communication".into())
    }
    fn finish(self, mut parts: PreparedExecutableParts<Run, ()>) -> Result<Run, String> {
        assert!(parts.take_communication().is_none());
        assert!(parts.take_processor().is_none());
        Ok(parts.into_executable())
    }
}

fn nemotron_prediction_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":7, "hidden_size":8,
        "intermediate_size":10, "num_hidden_layers":1,
        "hybrid_override_pattern":"*", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":2,
        "conv_kernel":3, "chunk_size":2, "n_routed_experts":2,
        "n_shared_experts":1, "moe_intermediate_size":4,
        "moe_shared_expert_intermediate_size":4, "num_experts_per_tok":1,
        "n_group":1, "topk_group":1, "num_nextn_predict_layers":1,
        "mtp_hybrid_override_pattern":"*", "tie_word_embeddings":false
    })
}

fn execute(head_scale: f32, change_extension: bool) -> Run {
    execute_config(
        nemotron_prediction_config(),
        head_scale,
        change_extension,
        "mtp.layers.0.eh_proj.weight",
    )
}

fn execute_config(
    config: serde_json::Value,
    head_scale: f32,
    change_extension: bool,
    extension_projection: &str,
) -> Run {
    let (root, mut expected) = prepared_adapter::payload_fixture_config(&config, head_scale);
    if change_extension {
        // Change one extension-only physical scalar before artifact inspection.
        // The resulting proposal must change while canonical target work cannot.
        let projection = expected.get_mut(extension_projection).unwrap();
        projection.1[0] = (f32::from_bits(projection.1[0]) + 0.5).to_bits();
        let encoded = expected
            .iter()
            .map(|(name, (shape, bits))| {
                (
                    name,
                    (
                        shape
                            .iter()
                            .map(|&value| value as usize)
                            .collect::<Vec<_>>(),
                        bits.iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let views = encoded
            .iter()
            .map(|(name, (shape, bytes))| {
                (
                    name.as_str(),
                    safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::tensor::serialize_to_file(views, None, &root.path().join("model.safetensors"))
            .unwrap();
    }
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let schema = inspection
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let mut aliases = schema
        .common_tensors
        .iter()
        .chain(
            schema
                .layout_groups
                .iter()
                .flat_map(|group| &group.variants)
                .flat_map(|variant| variant.tensors.iter()),
        )
        .filter(|constraint| expected.contains_key(&constraint.key))
        .map(|constraint| (constraint.key.clone(), constraint.aliases.clone()))
        .collect::<BTreeMap<_, _>>();
    let plan = prepared_adapter::plan(None).with_drafting(eredu_core::DraftingPlan::Embedded {
        max_draft_tokens: 1,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        &plan,
        eredu_runtime::ResidencyDiagnostics::new(false, false),
        None,
    )
    .unwrap();
    let selected =
        eredu_architectures::select_preparation(&inspection, &request, &Provider).unwrap();
    let admitted =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let sources =
        eredu_architectures::prepared_sources::prepare_model_sources(admitted, selected).unwrap();
    let target_keys = sources
        .target()
        .source_keys()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let extension_keys = sources
        .extension()
        .unwrap()
        .source_keys()
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert!(!target_keys.is_empty() && !extension_keys.is_empty());
    assert!(target_keys.is_disjoint(&extension_keys));
    assert!(sources
        .target()
        .source_metadata(extension_keys.first().unwrap())
        .is_err());
    assert!(sources
        .extension()
        .unwrap()
        .source_metadata(target_keys.first().unwrap())
        .is_err());
    let context = NumericContext {
        bind_checkpoint_values: true,
        ..NumericContext::default()
    };
    let snapshots = Rc::new(RefCell::new(Vec::new()));
    let materialize_snapshots = Rc::clone(&snapshots);
    reset_reference_stage_evidence("SafeTensors");
    let materialize = |prepared: PreparedPredictionExtension<NumericBackend>,
                       source: RetainedCheckpointSource| {
        assert_eq!(
            source.source_keys().into_iter().collect::<BTreeSet<_>>(),
            extension_keys
        );
        prepared.materialize::<Materializer>(&mut Materialization {
            source,
            context: &context,
            snapshots: materialize_snapshots,
        })
    };
    let visitor = |binding| Visitor {
        context: &context,
        binding,
        snapshots,
        target_keys,
        component_scopes: eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .resolve_safetensors(&config)
            .unwrap()
            .architecture_plan()
            .architecture_descriptor()
            .component_scopes,
    };
    let routed = config["num_experts"].as_i64().unwrap_or(0) > 0
        || (config["model_type"] == "nemotron_h"
            && config["hybrid_override_pattern"]
                .as_str()
                .unwrap()
                .contains('E'));
    let run = if routed {
        let route = RoutedRoute::<NumericBackend, State, State, _, _, _>::new(
            &context, &context, Ordinary, Ordinary, Ordinary,
        )
        .with_prediction::<Materializer, _, _>(materialize, visitor);
        construct_prepared_execution(
            sources,
            None,
            PreparedExecutionRoutes::new().with_routed(route),
            Assembler,
        )
        .unwrap()
    } else {
        let route = ReplicatedRoute::<NumericBackend, _>::new(
            &context,
            &context,
            SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(Ordinary),
        )
        .with_prediction::<Materializer, _, _>(materialize, visitor);
        construct_prepared_execution(
            sources,
            None,
            PreparedExecutionRoutes::new().with_replicated(route),
            Assembler,
        )
        .unwrap()
    };
    canonical_qwen_fixture(&config, &mut expected, &mut aliases);
    let evidence = last_reference_stage_evidence();
    let mut matched = BTreeSet::new();
    for (logical, value) in &evidence.bound_parameters {
        let sources = expected
            .keys()
            .filter(|source| {
                *source == logical
                    || aliases
                        .get(*source)
                        .is_some_and(|names| names.contains(logical))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sources.len(),
            1,
            "executed parameter has one declared checkpoint source: {logical}"
        );
        let source = sources[0];
        assert_eq!(
            value, &expected[source],
            "executed target/extension parameter {logical} equals source {source}"
        );
        assert!(
            matched.insert(source.clone()),
            "one physical scalar payload must not replace another parameter"
        );
    }
    assert_eq!(
        matched,
        expected.keys().cloned().collect(),
        "every target and extension checkpoint parameter must execute"
    );
    assert!(evidence
        .payload_reads
        .iter()
        .all(|read| read.physically_bounded && read.encoded_bytes > 0));
    run
}

pub(super) fn assert_real_embedded_prediction() {
    let ordinary = execute(1.0, false);
    let changed = execute(2.0, false);
    let changed_extension = execute(1.0, true);
    assert_prediction_payload_effects(ordinary, changed, changed_extension);
}

fn assert_prediction_payload_effects(ordinary: Run, changed: Run, changed_extension: Run) {
    assert!(ordinary
        .proposal
        .data
        .iter()
        .any(|value| value.abs() > 1e-7));
    assert_tensor_close(
        &changed.target,
        &ordinary.target.map(|value| 2.0 * value),
        "selected target head payload",
    );
    assert_tensor_close(
        &changed.proposal,
        &ordinary.proposal.map(|value| 2.0 * value),
        "prediction reuses exact target head payload",
    );
    assert_tensor_exact(
        &changed.capture,
        &ordinary.capture,
        "head does not alter target capture",
    );
    assert_tensor_exact(
        &changed.proposal_capture,
        &ordinary.proposal_capture,
        "head does not alter prediction capture",
    );
    assert_state_exact(
        &changed.target_state,
        &ordinary.target_state,
        ordinary.target_state.layout().len(),
        "head does not alter target state",
    );
    assert_state_exact(
        &changed.lane_state,
        &ordinary.lane_state,
        ordinary.lane_state.layout().len(),
        "head does not alter prediction state",
    );
    assert_tensor_exact(
        &changed_extension.target,
        &ordinary.target,
        "extension-only payload cannot alter target logits",
    );
    assert_tensor_exact(
        &changed_extension.capture,
        &ordinary.capture,
        "extension-only payload cannot alter target capture",
    );
    assert_state_exact(
        &changed_extension.target_state,
        &ordinary.target_state,
        ordinary.target_state.layout().len(),
        "extension-only payload cannot alter canonical cache",
    );
    assert_ne!(
        changed_extension.proposal.data, ordinary.proposal.data,
        "real prediction consumes its changed extension payload"
    );
}

#[test]
fn nemotron_prepared_prediction_components_cover_multiple_physical_units_and_depths() {
    for routed_target in [false, true] {
        for pattern in ["*E", "*E*E"] {
            for depth in [1, 2] {
                let mut config = nemotron_prediction_config();
                if routed_target {
                    config["num_hidden_layers"] = 2.into();
                    config["hybrid_override_pattern"] = "*E".into();
                }
                config["mtp_hybrid_override_pattern"] = pattern.into();
                config["num_nextn_predict_layers"] = depth.into();
                config["moe_intermediate_size"] = 6.into();
                config["moe_shared_expert_intermediate_size"] = 8.into();
                let ordinary =
                    execute_config(config.clone(), 1.0, false, "mtp.layers.0.eh_proj.weight");
                let head =
                    execute_config(config.clone(), 2.0, false, "mtp.layers.0.eh_proj.weight");
                let fusion = execute_config(config, 1.0, true, "mtp.layers.0.eh_proj.weight");
                assert_prediction_payload_effects(ordinary, head, fusion);
            }
        }
    }
}

#[test]
fn embedded_prediction_binds_real_payload_and_restores_exact_lane_state() {
    super::run_reference_conformance_embedded_prediction();
}

#[test]
fn qwen_text_prepared_prediction_binds_shared_fusion_and_restores_lane_state() {
    for mut config in heterogeneous_replicated_configs()
        .into_iter()
        .filter(|config| {
            matches!(
                config["model_type"].as_str(),
                Some("qwen3_next" | "qwen3_5_text")
            )
        })
        .flat_map(|config| {
            let mut routed = config.clone();
            if routed["model_type"] == "qwen3_5_text" {
                routed["model_type"] = "qwen3_5_moe_text".into();
            }
            routed["num_experts"] = 4.into();
            routed["num_experts_per_tok"] = 2.into();
            routed["moe_intermediate_size"] = 6.into();
            routed["shared_expert_intermediate_size"] = 8.into();
            routed["norm_topk_prob"] = true.into();
            [config, routed]
        })
    {
        for depth in [1, 2] {
            config["mtp_num_hidden_layers"] = depth.into();
            config["tie_word_embeddings"] = false.into();
            let ordinary = execute_config(config.clone(), 1.0, false, "mtp.fc.weight");
            let head = execute_config(config.clone(), 2.0, false, "mtp.fc.weight");
            let fusion = execute_config(config.clone(), 1.0, true, "mtp.fc.weight");
            assert_prediction_payload_effects(ordinary, head, fusion);
        }
    }
}

// Independent scalar indexing of the released Next group-major fused rows.
// The preparation binder instead executes the architecture's selected recipes.
fn canonical_qwen_fixture(
    config: &serde_json::Value,
    expected: &mut BTreeMap<String, (Vec<i32>, Vec<u32>)>,
    aliases: &mut BTreeMap<String, Vec<String>>,
) {
    // Independent scalar packing of per-expert source matrices. The real binder
    // consumes its retained concatenation/stack recipes instead of this map.
    let roots = expected
        .keys()
        .filter_map(|name| name.strip_suffix(".0.down_proj.weight"))
        .filter(|root| root.ends_with(".experts"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let experts = config["num_experts"].as_u64().unwrap_or(0) as usize;
    for root in roots {
        assert!(experts > 0);
        let mut packed_gate_up = Vec::new();
        let mut packed_down = Vec::new();
        let mut gate_shape = vec![];
        let mut down_shape = vec![];
        for expert in 0..experts {
            for field in ["gate_proj", "up_proj", "down_proj"] {
                let name = format!("{root}.{expert}.{field}.weight");
                let (shape, bits) = expected.remove(&name).unwrap();
                aliases.remove(&name);
                assert_eq!(shape.len(), 2);
                if field == "down_proj" {
                    if down_shape.is_empty() {
                        down_shape = shape.clone();
                    }
                    assert_eq!(shape, down_shape);
                    packed_down.extend(bits);
                } else {
                    if gate_shape.is_empty() {
                        gate_shape = shape.clone();
                    }
                    assert_eq!(shape, gate_shape);
                    packed_gate_up.extend(bits);
                }
            }
        }
        expected.insert(
            format!("{root}.gate_up_proj"),
            (
                vec![experts as i32, 2 * gate_shape[0], gate_shape[1]],
                packed_gate_up,
            ),
        );
        expected.insert(
            format!("{root}.down_proj"),
            (
                vec![experts as i32, down_shape[0], down_shape[1]],
                packed_down,
            ),
        );
    }
    if config["model_type"] != "qwen3_next" {
        return;
    }
    let groups = config["linear_num_key_heads"].as_u64().unwrap() as usize;
    let key = config["linear_key_head_dim"].as_u64().unwrap() as usize;
    let values_per_group = config["linear_num_value_heads"].as_u64().unwrap() as usize / groups;
    let value = values_per_group * config["linear_value_head_dim"].as_u64().unwrap() as usize;
    let sources = expected
        .keys()
        .filter(|name| {
            name.ends_with(".in_proj_qkvz.weight") || name.ends_with(".in_proj_ba.weight")
        })
        .cloned()
        .collect::<Vec<_>>();
    for source in sources {
        let (shape, data) = expected.remove(&source).unwrap();
        let columns = shape[1] as usize;
        let (suffix, widths, outputs) = if source.ends_with("in_proj_qkvz.weight") {
            (
                "in_proj_qkvz.weight",
                vec![key, key, value, value],
                vec![
                    ("in_proj_qkv.weight", vec![0, 1, 2]),
                    ("in_proj_z.weight", vec![3]),
                ],
            )
        } else {
            (
                "in_proj_ba.weight",
                vec![values_per_group, values_per_group],
                vec![("in_proj_b.weight", vec![0]), ("in_proj_a.weight", vec![1])],
            )
        };
        let stride: usize = widths.iter().sum();
        assert_eq!(shape[0] as usize, groups * stride);
        for (target, parts) in outputs {
            let mut selected = Vec::new();
            for part in parts {
                let offset: usize = widths[..part].iter().sum();
                for group in 0..groups {
                    let start = (group * stride + offset) * columns;
                    selected.extend_from_slice(&data[start..start + widths[part] * columns]);
                }
            }
            let name = format!("{}{target}", source.strip_suffix(suffix).unwrap());
            let rows = (selected.len() / columns) as i32;
            expected.insert(name.clone(), (vec![rows, shape[1]], selected));
            aliases.insert(name, vec![]);
        }
    }
}
