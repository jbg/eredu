//! Real scalar embedded prediction through retained preparation and the common driver.

use super::*;
use eredu_architectures::{
    prediction_extension::*,
    prepared_execution::*,
    replicated_text::{ReplicatedPredictionProfileDispatcher, ReplicatedPredictionTargetVisitor},
    PreparationMechanismProvider,
};
use std::rc::Rc;

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
    source: SharedCheckpointSource,
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
            Agreement,
            Publication,
        ])
    }
    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        eredu_runtime::CommunicationCapabilities::new([]).unwrap()
    }
}

struct Invoker<'a, A>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
{
    session: &'a mut eredu_runtime::ReplicatedTextSession<
        A,
        NumericBackend,
        NumericReplicatedMechanisms,
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
impl<A> PredictionOperationInvoker<A, NumericBackend, State> for Invoker<'_, A>
where
    A: LayeredArchitecture<NumericBackend, State, Error = Error>,
    A::StaticModules: Clone,
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
        mut extension: <A as MaterializedPredictionTarget<NumericBackend>>::Extension<Materializer>,
        checkpoint: SharedCheckpointSource,
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
        let (mut session, _) = construct_selected_text_session::<NumericBackend, _, _>(
            prepared,
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint),
            self.context,
        )?;
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
        assert_eq!(extension.depth(), 1);
        let mut lane = extension.new_state();
        let mut invoker = Invoker {
            session: &mut session,
            context: self.context,
            fail_after: false,
        };
        extension.prefill::<State, _>(&mut invoker, &capture, &capture, &tokens, &mut lane)?;
        let checkpoint = lane.clone();
        let prior = capture.axis_slice(1, 2, 3);
        let token = NumericTensor::token_ids(&[4]);
        let (proposal, proposal_capture) =
            extension.logits::<State, _>(&mut invoker, &prior, &token, 0, &mut lane)?;
        let _ = lane.clone();
        let expected_lane = self.snapshots.borrow().last().unwrap().clone();
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
        assert!(lane_state
            .as_ref()
            .iter()
            .all(|layer| layer.position() == 4));
        assert!(target_state
            .as_ref()
            .iter()
            .all(|layer| layer.position() == 3));
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

struct Ordinary;
impl ReplicatedTextArchitectureVisitor<NumericBackend, State> for Ordinary {
    type Output = Run;
    type Error = String;
    fn construction_started(&mut self) {}
    fn visit<A>(
        self,
        _: PreparedReplicatedTextArchitecture<A>,
        _: SharedCheckpointSource,
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

fn execute(head_scale: f32, change_extension: bool) -> Run {
    let config = serde_json::json!({
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
    });
    let (root, mut expected) = prepared_adapter::payload_fixture_config(&config, head_scale);
    if change_extension {
        // Change one extension-only physical scalar before artifact inspection.
        // The resulting proposal must change while canonical target work cannot.
        let projection = expected.get_mut("mtp.layers.0.eh_proj.weight").unwrap();
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
    let aliases = schema
        .common_tensors
        .iter()
        .chain(
            schema
                .layout_groups
                .iter()
                .filter(|group| group.required)
                .filter_map(|group| group.variants.first())
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
    let route = ReplicatedRoute::<NumericBackend, _>::new(
        &context,
        &context,
        SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(Ordinary),
    )
    .with_prediction::<Materializer, _, _>(
        |prepared: PreparedPredictionExtension<NumericBackend>, source: SharedCheckpointSource| {
            assert_eq!(
                source.source_keys().into_iter().collect::<BTreeSet<_>>(),
                extension_keys
            );
            prepared.materialize::<Materializer>(&mut Materialization {
                source,
                context: &context,
                snapshots: materialize_snapshots,
            })
        },
        |binding| Visitor {
            context: &context,
            binding,
            snapshots,
            target_keys,
        },
    );
    let run = construct_prepared_execution(
        sources,
        None,
        PreparedExecutionRoutes::new().with_replicated(route),
        Assembler,
    )
    .unwrap();
    let evidence = last_reference_stage_evidence();
    let mut matched = BTreeSet::new();
    for (logical, value) in &evidence.bound_parameters {
        let sources = aliases
            .iter()
            .filter(|(source, aliases)| *source == logical || aliases.contains(logical))
            .collect::<Vec<_>>();
        assert_eq!(
            sources.len(),
            1,
            "executed parameter has one declared checkpoint source: {logical}"
        );
        let source = sources[0].0;
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
fn embedded_prediction_binds_real_payload_and_restores_exact_lane_state() {
    super::run_reference_conformance_embedded_prediction();
}
