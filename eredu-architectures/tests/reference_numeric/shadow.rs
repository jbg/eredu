//! A non-MLX client/device/completion adapter over real scalar model execution.
//!
//! Runtime labels describe hypothetical capability sets, not implementations of
//! those runtimes. Tensor and layer dispatch remains the scalar backend's static
//! dispatch; only this outer model interface is erased.

use super::*;
use eredu_architectures::{prepared_execution::*, PreparationMechanismProvider};
use eredu_runtime::ReplicatedTextMechanismSupport;
use std::rc::Rc;

#[path = "shadow/provider.rs"]
mod provider;

type State = DeviceState<NumericBackend, NumericHybridLayerState>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeShape {
    Cpu,
    Wgpu,
    Cuda,
    Hip,
}

#[derive(Default, Debug)]
struct Counts {
    normalization: Cell<usize>,
    capabilities: Cell<usize>,
    candidates: Cell<usize>,
    selection: Cell<usize>,
    sources: Cell<usize>,
    native: Cell<usize>,
    construction: Cell<usize>,
    publication: Cell<usize>,
    submissions: Cell<usize>,
    completions: Cell<usize>,
    drains: Cell<usize>,
    client_drops: Cell<usize>,
}
fn increment(value: &Cell<usize>) {
    value.set(value.get() + 1);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Device {
    runtime: RuntimeShape,
    ordinal: u32,
}
struct Client {
    device: Device,
    owner: Rc<()>,
    counts: Rc<Counts>,
}
impl Drop for Client {
    fn drop(&mut self) {
        increment(&self.counts.client_drops);
    }
}

struct Support {
    runtime: RuntimeShape,
    counts: Rc<Counts>,
}
impl ReplicatedTextMechanismSupport for Support {
    fn facts(
        &self,
        _: &eredu_runtime::CacheResidencyPolicy,
    ) -> eredu_runtime::BackendMechanismFacts {
        let residencies = if matches!(self.runtime, RuntimeShape::Wgpu) {
            vec![eredu_runtime::WeightResidencyMechanism::Resident]
        } else {
            vec![
                eredu_runtime::WeightResidencyMechanism::Resident,
                eredu_runtime::WeightResidencyMechanism::Windowed,
                eredu_runtime::WeightResidencyMechanism::DiskStreamed,
            ]
        };
        eredu_runtime::BackendMechanismFacts::new(
            NumericBackend::OPERATOR_CAPABILITIES,
            residencies,
            eredu_runtime::StateLifecycleCapabilities::new()
                .with_transactions(true, true)
                .with_reset(true)
                .with_prompt_cache(true)
                .with_observation_retention(true),
        )
        .with_session(eredu_core::SessionCapabilities::new(true, true, false))
        .with_prompt_cache(true)
        .with_exact_completion(true)
    }
    fn supports_direct(&self, descriptor: &eredu_runtime::WeightLoweringDescriptor) -> bool {
        increment(&self.counts.candidates);
        if descriptor.executable() != eredu_checkpoint::LinearFormat::Dense {
            return false;
        }
        match descriptor.source().scalar_dtype() {
            Some(eredu_checkpoint::StoredDtype::F32) => true,
            Some(eredu_checkpoint::StoredDtype::F16) => {
                matches!(self.runtime, RuntimeShape::Cuda | RuntimeShape::Hip)
            }
            _ => false,
        }
    }
    fn supports_transform(&self, descriptor: &eredu_runtime::WeightLoweringDescriptor) -> bool {
        increment(&self.counts.candidates);
        matches!(self.runtime, RuntimeShape::Hip)
            && descriptor.source().scalar_dtype() == Some(eredu_checkpoint::StoredDtype::F32)
            && matches!(descriptor.executable(), eredu_checkpoint::LinearFormat::Affine(format) if format.bits == 4 && format.group_size == 16)
    }
    fn supports_state_component(
        &self,
        component: &eredu_core::cache::StateComponentPolicy,
        placement: eredu_runtime::StateComponentPlacement,
    ) -> bool {
        placement == eredu_runtime::StateComponentPlacement::Device
            && (!matches!(self.runtime, RuntimeShape::Wgpu)
                || !matches!(
                    component.role(),
                    eredu_core::cache::StateComponentRole::Fixed(_)
                ))
    }
}
impl PreparationMechanismProvider for Support {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        increment(&self.counts.capabilities);
        eredu_core::PreparationMechanismCapabilities::new(true, true)
            .with_safetensors_quantization(matches!(self.runtime, RuntimeShape::Hip), false)
            .with_input_modalities(eredu_core::InputModalities {
                text: true,
                image: false,
                audio: false,
                video: false,
            })
            .with_exact_completion(true)
            .with_session(eredu_core::SessionCapabilities::new(true, true, false))
            .with_residency(eredu_core::ResidencyRequest::FullyResident, true)
            .with_residency(
                eredu_core::ResidencyRequest::LayerwiseHost,
                !matches!(self.runtime, RuntimeShape::Wgpu),
            )
            .with_residency(
                eredu_core::ResidencyRequest::DenseDiskStream,
                !matches!(self.runtime, RuntimeShape::Wgpu),
            )
    }
    fn supports_grouped_operation(&self, _: eredu_runtime::GroupedOperationRequirement) -> bool {
        false
    }
    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        eredu_runtime::synthesize_replicated_text_capabilities(requirements, request, self)
    }
    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        eredu_runtime::MediaPrimitiveCapabilities::new([], [], [], [], 1)
    }
    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        eredu_runtime::SpeculativeMechanismCapabilities::new([])
    }
    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        eredu_runtime::CommunicationCapabilities::new([]).unwrap()
    }
}

trait Body {
    fn forward(
        &mut self,
        tokens: &NumericTensor,
        prefill: bool,
        context: &NumericContext,
    ) -> Result<NumericTensor, String>;
    fn checkpoint(&mut self, context: &NumericContext) -> Result<State, String>;
    fn rollback(&mut self, state: State, context: &NumericContext) -> Result<(), String>;
    fn reset(&mut self, context: &NumericContext) -> Result<(), String>;
    fn state(&self) -> Result<State, String>;
}
impl<A> Body
    for eredu_runtime::ReplicatedTextSession<A, NumericBackend, NumericReplicatedMechanisms>
where
    A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error> + 'static,
{
    fn forward(
        &mut self,
        tokens: &NumericTensor,
        prefill: bool,
        context: &NumericContext,
    ) -> Result<NumericTensor, String> {
        if prefill {
            self.prefill(tokens, None, context)
        } else {
            self.decode(tokens, context)
        }
        .map_err(|error| error.to_string())
    }
    fn checkpoint(&mut self, context: &NumericContext) -> Result<State, String> {
        self.checkpoint(context).map_err(|error| error.to_string())
    }
    fn rollback(&mut self, state: State, context: &NumericContext) -> Result<(), String> {
        self.rollback(state, context)
            .map_err(|error| error.to_string())
    }
    fn reset(&mut self, context: &NumericContext) -> Result<(), String> {
        self.reset(context).map_err(|error| error.to_string())
    }
    fn state(&self) -> Result<State, String> {
        self.report()
            .map(|report| report.state_report().clone())
            .map_err(|error| error.to_string())
    }
}

struct Visitor<'a> {
    context: &'a NumericContext,
    counts: Rc<Counts>,
}
impl ReplicatedTextArchitectureVisitor<NumericBackend, State> for Visitor<'_> {
    type Output = Box<dyn Body>;
    type Error = String;
    fn construction_started(&mut self) {
        increment(&self.counts.construction);
    }
    fn visit<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        checkpoint: SharedCheckpointSource,
    ) -> Result<Self::Output, String>
    where
        A: eredu_runtime::ReplicatedTextArchitecture<NumericBackend, State, Error = Error>
            + 'static,
        A::StaticModules: Clone,
        A::Error: std::fmt::Display,
    {
        let mechanisms = if prepared.selected().residency().is_fully_resident() {
            NumericReplicatedMechanisms::with_bound_checkpoint(checkpoint)
        } else {
            NumericReplicatedMechanisms::with_bounded_checkpoint(checkpoint)
        };
        construct_selected_text_session(prepared, mechanisms, self.context)
            .map(|(session, _facts)| Box::new(session) as Box<dyn Body>)
    }
}

struct Assembler {
    client: Rc<Client>,
    context: NumericContext,
    fail_publication: bool,
}
impl PreparedExecutableAssembler<()> for Assembler {
    type Executable = Box<dyn Body>;
    type Output = Session;
    type Error = String;
    fn floating_state_bytes(
        &mut self,
        _: &eredu_architectures::preparation::FloatingStateDtypeSource,
    ) -> Result<std::num::NonZeroU8, String> {
        Ok(std::num::NonZeroU8::new(4).unwrap())
    }
    fn validate_communication(&mut self, _: &CommunicationManifest, _: &()) -> Result<(), String> {
        Err("scalar shadow has no distributed mechanism".into())
    }
    fn finish(
        self,
        mut parts: PreparedExecutableParts<Box<dyn Body>, ()>,
    ) -> Result<Session, String> {
        if parts.take_processor().is_some() || parts.take_communication().is_some() {
            return Err("unimplemented optional native resource".into());
        }
        eredu_core::SessionAdmission::new(parts.capabilities())
            .validate(eredu_core::SessionCapabilities::new(true, true, false))
            .map_err(|error| error.to_string())?;
        if self.fail_publication {
            return Err("injected publication failure".into());
        }
        increment(&self.client.counts.publication);
        Ok(Session {
            body: Rc::new(RefCell::new(parts.into_executable())),
            context: self.context,
            client: self.client,
            authority: eredu_core::SessionAuthority::new(),
            reported_capabilities: eredu_core::SessionCapabilities::new(true, true, false),
        })
    }
}

fn load(
    inspection: &eredu_core::ArtifactInspection<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    plan: &eredu_core::ExecutionPlan,
    runtime: RuntimeShape,
    counts: Rc<Counts>,
    fail_publication: bool,
) -> Result<Session, String> {
    increment(&counts.normalization);
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        plan,
        eredu_runtime::ResidencyDiagnostics::default(),
        None,
    )
    .map_err(|error| error.to_string())?;
    let backend = provider::Provider::new(runtime, Rc::clone(&counts), fail_publication);
    eredu_core::prepare_inspected_model(&backend, inspection.clone(), (request, backend.target()))
        .map(eredu_core::PreparedModel::into_inner)
        .map_err(|error| error.to_string())
}

fn materialize(
    sources: eredu_architectures::prepared_sources::PreparedModelSources,
    device: Device,
    owner: Rc<()>,
    counts: Rc<Counts>,
    fail_publication: bool,
) -> Result<Session, String> {
    increment(&counts.native);
    let client = Rc::new(Client {
        device,
        owner,
        counts: Rc::clone(&counts),
    });
    let context = NumericContext::default();
    let routes =
        PreparedExecutionRoutes::new().with_replicated(ReplicatedRoute::<NumericBackend, _>::new(
            &context,
            &context,
            SharedReplicatedTextVisitor::<NumericReplicatedStateProfiles, _>::new(Visitor {
                context: &context,
                counts,
            }),
        ));
    construct_prepared_execution(
        sources,
        None,
        routes,
        Assembler {
            client,
            context: context.clone(),
            fail_publication,
        },
    )
    .map_err(|error| error.to_string())
}

struct Session {
    body: Rc<RefCell<Box<dyn Body>>>,
    context: NumericContext,
    client: Rc<Client>,
    authority: eredu_core::SessionAuthority,
    reported_capabilities: eredu_core::SessionCapabilities,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum EventState {
    Pending,
    Ready,
    Failed,
}
struct NativeCompletion {
    state: Rc<Cell<EventState>>,
    body: Rc<RefCell<Box<dyn Body>>>,
    rollback: RefCell<Option<State>>,
    context: NumericContext,
    client: Rc<Client>,
    lease: eredu_core::SubmissionLease,
}
impl Completion for NativeCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Error> {
        match self.state.get() {
            EventState::Pending => Ok(false),
            EventState::Ready => {
                self.rollback.borrow_mut().take();
                if self.lease.resolve() {
                    increment(&self.client.counts.completions);
                }
                Ok(true)
            }
            EventState::Failed => {
                if let Some(state) = self.rollback.borrow_mut().take() {
                    self.body
                        .borrow_mut()
                        .rollback(state, &self.context)
                        .map_err(Error::backend)?;
                }
                if self.lease.resolve() {
                    increment(&self.client.counts.completions);
                }
                Err(Error::backend("terminal scalar queue failure"))
            }
        }
    }
    fn wait(&self) -> Result<(), Error> {
        if self.is_complete()? {
            Ok(())
        } else {
            Err(Error::backend("scalar queue is still pending"))
        }
    }
}
impl Drop for NativeCompletion {
    fn drop(&mut self) {
        if self.state.get() == EventState::Pending {
            // This synthetic queue supports safe cancellation after synchronous
            // numeric execution; rollback precedes release of mutable authority.
            self.state.set(EventState::Failed);
            increment(&self.client.counts.drains);
        }
        let _ = self.is_complete();
    }
}
impl Session {
    fn submit(
        &mut self,
        tokens: &NumericTensor,
        prefill: bool,
    ) -> Result<eredu_core::Submission<NumericTensor, NativeCompletion>, String> {
        let lease = self
            .authority
            .begin_submission()
            .map_err(|error| error.to_string())?;
        increment(&self.client.counts.submissions);
        let checkpoint = self.body.borrow_mut().checkpoint(&self.context)?;
        let output = self
            .body
            .borrow_mut()
            .forward(tokens, prefill, &self.context)?;
        let state = if matches!(self.client.device.runtime, RuntimeShape::Cpu) {
            EventState::Ready
        } else {
            EventState::Pending
        };
        Ok(eredu_core::Submission {
            output,
            completion: NativeCompletion {
                state: Rc::new(Cell::new(state)),
                body: Rc::clone(&self.body),
                rollback: RefCell::new(Some(checkpoint)),
                context: self.context.clone(),
                client: Rc::clone(&self.client),
                lease,
            },
        })
    }
    fn reset(&mut self) -> Result<(), String> {
        self.authority
            .require_idle()
            .map_err(|error| error.to_string())?;
        self.body.borrow_mut().reset(&self.context)
    }
    fn checkpoint(&mut self) -> Result<State, String> {
        self.authority
            .require_idle()
            .map_err(|error| error.to_string())?;
        self.body.borrow_mut().checkpoint(&self.context)
    }
    fn rollback(&mut self, state: State) -> Result<(), String> {
        self.authority
            .require_idle()
            .map_err(|error| error.to_string())?;
        self.body.borrow_mut().rollback(state, &self.context)
    }
}

#[test]
fn real_scalar_session_preserves_pending_failure_drop_and_client_lifetimes() {
    let (root, expected_parameters) = prepared_adapter::payload_fixture(1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let mut expected_output = None;
    for runtime in [
        RuntimeShape::Cpu,
        RuntimeShape::Wgpu,
        RuntimeShape::Cuda,
        RuntimeShape::Hip,
    ] {
        let counts = Rc::new(Counts::default());
        reset_reference_stage_evidence("SafeTensors");
        let mut session = load(
            &inspection,
            &prepared_adapter::plan(None),
            runtime,
            Rc::clone(&counts),
            false,
        )
        .unwrap();
        assert_eq!(session.client.device.ordinal, 0);
        assert_eq!(
            last_reference_stage_evidence().bound_parameters,
            expected_parameters
        );
        assert_eq!(
            (
                counts.normalization.get(),
                counts.selection.get(),
                counts.sources.get(),
                counts.native.get(),
                counts.construction.get(),
                counts.publication.get()
            ),
            (1, 1, 1, 1, 1, 1)
        );
        assert!(counts.capabilities.get() > 0 && counts.candidates.get() > 0);
        let before = session.checkpoint().unwrap();
        let first = session
            .submit(&NumericTensor::token_ids(&[1, 3, 2]), true)
            .unwrap();
        match &expected_output {
            None => expected_output = Some(first.output.clone()),
            Some(expected) => {
                assert_tensor_exact(&first.output, expected, "capability-shape numerical output")
            }
        }
        // Even ready CPU work owns authority until its completion is observed.
        assert!(session.reset().is_err());
        assert!(session.checkpoint().is_err());
        assert!(session.rollback(before.clone()).is_err());
        assert!(session
            .submit(&NumericTensor::token_ids(&[4]), false)
            .is_err());
        assert_eq!(counts.submissions.get(), 1);
        if !matches!(runtime, RuntimeShape::Cpu) {
            assert!(!first.completion.is_complete().unwrap());
            assert!(first.completion.wait().is_err());
            assert!(session.reset().is_err());
        }
        first.completion.state.set(EventState::Ready);
        first.completion.wait().unwrap();
        assert_eq!(counts.completions.get(), 1);
        let committed = session.checkpoint().unwrap();

        let failed = session
            .submit(&NumericTensor::token_ids(&[4]), false)
            .unwrap();
        failed.completion.state.set(EventState::Pending);
        first.completion.wait().unwrap();
        assert!(
            session.reset().is_err(),
            "stale completion cannot release the newer ticket"
        );
        failed.completion.state.set(EventState::Failed);
        assert!(failed.completion.wait().is_err());
        assert_state_exact(
            &session.body.borrow().state().unwrap(),
            &committed,
            committed.layout().len(),
            "terminal native failure rolls back before release",
        );
        let dropped = session
            .submit(&NumericTensor::token_ids(&[5]), false)
            .unwrap();
        dropped.completion.state.set(EventState::Pending);
        drop(dropped);
        assert_state_exact(
            &session.checkpoint().unwrap(),
            &committed,
            committed.layout().len(),
            "pending Drop safely cancels before release",
        );
        assert_eq!(counts.drains.get(), 1);
        drop(first);
        drop(failed);
        let survivor = session
            .submit(&NumericTensor::token_ids(&[6]), false)
            .unwrap();
        survivor.completion.state.set(EventState::Pending);
        drop(session);
        assert_eq!(
            counts.client_drops.get(),
            0,
            "pending completion retains native client after session drop"
        );
        survivor.completion.state.set(EventState::Ready);
        survivor.completion.wait().unwrap();
        drop(survivor);
        assert_eq!(counts.client_drops.get(), 1);
        assert_eq!(counts.submissions.get(), counts.completions.get());
    }
}

#[test]
fn unsupported_requests_fail_before_source_payload_and_native_work() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    for plan in [
        prepared_adapter::plan(None).with_max_cached_shards(0),
        prepared_adapter::plan(Some(eredu_core::QuantizationRequest::Affine {
            bits: 4,
            group_size: 16,
        })),
        prepared_adapter::plan(None).with_residency(eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: None,
            host_budget_bytes: None,
        }),
    ] {
        let counts = Rc::new(Counts::default());
        reset_reference_stage_evidence("SafeTensors");
        assert!(load(
            &inspection,
            &plan,
            RuntimeShape::Wgpu,
            Rc::clone(&counts),
            false
        )
        .is_err());
        assert_eq!(counts.normalization.get(), 1);
        assert_eq!(
            (
                counts.sources.get(),
                counts.native.get(),
                counts.construction.get(),
                counts.publication.get()
            ),
            (0, 0, 0, 0)
        );
        assert!(last_reference_stage_evidence().payload_reads.is_empty());
    }

    let mixed = heterogeneous_replicated_configs()
        .into_iter()
        .find(|config| config["model_type"] == "lfm2")
        .unwrap();
    let (root, _) = prepared_adapter::payload_fixture_config(&mixed, 1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let counts = Rc::new(Counts::default());
    assert!(load(
        &inspection,
        &prepared_adapter::plan(None),
        RuntimeShape::Wgpu,
        Rc::clone(&counts),
        false
    )
    .is_err());
    assert!(
        counts.candidates.get() > 0,
        "fixed-state denial occurs during exact capability synthesis"
    );
    assert_eq!(
        (
            counts.sources.get(),
            counts.native.get(),
            counts.construction.get()
        ),
        (0, 0, 0)
    );
}

#[test]
fn failed_publication_drops_materialized_body_and_native_client() {
    let (root, _) = prepared_adapter::payload_fixture(1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let counts = Rc::new(Counts::default());
    assert!(load(
        &inspection,
        &prepared_adapter::plan(None),
        RuntimeShape::Cuda,
        Rc::clone(&counts),
        true
    )
    .is_err());
    assert_eq!(
        (
            counts.sources.get(),
            counts.native.get(),
            counts.construction.get(),
            counts.publication.get(),
            counts.submissions.get(),
            counts.client_drops.get()
        ),
        (1, 1, 1, 0, 0, 1)
    );
}

#[test]
fn bounded_scalar_binding_reads_one_selected_unit_at_a_time_and_matches_resident() {
    let (root, expected_parameters) = prepared_adapter::payload_fixture(1.0);
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let mut expected = None;
    for residency in [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(1 << 20),
            host_budget_bytes: Some(1 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 1 << 20,
            host_budget_bytes: 1 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let bounded = !matches!(residency, eredu_core::ResidencyPlan::FullyResident);
        reset_reference_stage_evidence("SafeTensors");
        let plan = prepared_adapter::plan(None).with_residency(residency);
        let mut session = load(
            &inspection,
            &plan,
            RuntimeShape::Cpu,
            Rc::new(Counts::default()),
            false,
        )
        .unwrap();
        let before = last_reference_stage_evidence();
        assert!(before.bounded_unit_acquisitions.is_empty());
        if bounded {
            assert!(
                before.bound_parameters.len() < expected_parameters.len(),
                "bounded construction must leave unit payloads lazy"
            );
        }
        let mut outputs = Vec::new();
        for (index, tokens) in [vec![1, 3, 2], vec![4], vec![5]].into_iter().enumerate() {
            let submission = session
                .submit(&NumericTensor::token_ids(&tokens), index == 0)
                .unwrap();
            submission.completion.wait().unwrap();
            outputs.push(submission.output);
        }
        let state = session.checkpoint().unwrap();
        let evidence = last_reference_stage_evidence();
        assert_eq!(evidence.bound_parameters, expected_parameters);
        if bounded {
            assert_eq!(evidence.peak_bound_units, 1);
            let order = (0..state.layout().len()).collect::<Vec<_>>().repeat(3);
            assert_eq!(evidence.bounded_unit_acquisitions, order);
            assert!(evidence.payload_reads.len() > before.payload_reads.len());
        }
        match &expected {
            None => expected = Some((outputs, state)),
            Some((expected_outputs, expected_state)) => {
                for (actual, expected) in outputs.iter().zip(expected_outputs) {
                    assert_tensor_exact(actual, expected, "bounded actual-payload continuation");
                }
                assert_state_exact(
                    &state,
                    expected_state,
                    state.layout().len(),
                    "bounded retained state",
                );
            }
        }
    }
}

#[test]
fn transformed_scalar_payload_matches_independently_expanded_dense_oracle() {
    use safetensors::tensor::{serialize_to_file, TensorView};

    let mut config = config("llama", false);
    config["hidden_size"] = 16.into();
    config["head_dim"] = 8.into();
    config["intermediate_size"] = 32.into();
    let (root, original) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let (oracle_root, _) = prepared_adapter::payload_fixture_config(&config, 1.0);
    let original_inspection =
        eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let plan = prepared_adapter::plan(Some(eredu_core::QuantizationRequest::Affine {
        group_size: 16,
        bits: 4,
    }));
    let request = eredu_runtime::NormalizedLoadRequest::from_execution_plan(
        &plan,
        eredu_runtime::ResidencyDiagnostics::default(),
        None,
    )
    .unwrap();
    let selected = eredu_architectures::select_preparation(
        &original_inspection,
        &request,
        &Support {
            runtime: RuntimeShape::Hip,
            counts: Rc::new(Counts::default()),
        },
    )
    .unwrap();
    let transformed_names = selected
        .text_realization()
        .materialization_tasks()
        .iter()
        .filter(|task| {
            matches!(
                task.lowering(),
                eredu_runtime::WeightLoweringKind::Transform
            )
        })
        .map(|task| task.name().to_owned())
        .collect::<BTreeSet<_>>();
    assert!(!transformed_names.is_empty());

    // Independent scalar affine oracle: nearest one of 16 equally spaced values
    // in each group. The backend materializer chooses integer codes; this oracle
    // instead searches the grid, so a rounding/companion error is observable.
    let mut expanded = BTreeMap::new();
    for (name, (shape, bits)) in &original {
        let mut values = bits
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        if transformed_names.contains(name) {
            for group in values.as_chunks_mut::<16>().0 {
                let low = group.iter().copied().reduce(f32::min).unwrap();
                let high = group.iter().copied().reduce(f32::max).unwrap();
                let step = (high - low) / 15.0;
                for value in group {
                    *value = (0..16)
                        .map(|code| code as f32 * step + low)
                        .min_by(|left, right| {
                            (left - *value).abs().total_cmp(&(right - *value).abs())
                        })
                        .unwrap();
                }
            }
        }
        expanded.insert(
            name.clone(),
            (
                shape.iter().map(|n| *n as usize).collect::<Vec<_>>(),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        );
    }
    let views = expanded
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &oracle_root.path().join("model.safetensors")).unwrap();
    let oracle_inspection =
        eredu_architectures::configuration::inspect_artifact(oracle_root.path()).unwrap();
    reset_reference_stage_evidence("SafeTensors");
    let mut transformed = load(
        &original_inspection,
        &plan,
        RuntimeShape::Hip,
        Rc::new(Counts::default()),
        false,
    )
    .unwrap();
    let bound = last_reference_stage_evidence().bound_parameters;
    assert!(
        bound.len() > original.len(),
        "selected affine companions are actually bound"
    );
    for (name, (shape, bytes)) in &expanded {
        let (bound_shape, bits) = &bound[name];
        assert_eq!(
            bound_shape.iter().map(|n| *n as usize).collect::<Vec<_>>(),
            *shape
        );
        let expected = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        assert_eq!(*bits, expected, "exact transformed parameter {name}");
    }
    let mut oracle = load(
        &oracle_inspection,
        &prepared_adapter::plan(None),
        RuntimeShape::Cpu,
        Rc::new(Counts::default()),
        false,
    )
    .unwrap();
    for (tokens, prefill) in [(&[1, 3, 2][..], true), (&[4][..], false), (&[5][..], false)] {
        let actual = transformed
            .submit(&NumericTensor::token_ids(tokens), prefill)
            .unwrap();
        actual.completion.state.set(EventState::Ready);
        actual.completion.wait().unwrap();
        let expected = oracle
            .submit(&NumericTensor::token_ids(tokens), prefill)
            .unwrap();
        expected.completion.wait().unwrap();
        assert_tensor_exact(
            &actual.output,
            &expected.output,
            "transformed versus expanded scalar logits",
        );
        let expected_state = oracle.checkpoint().unwrap();
        assert_state_exact(
            &transformed.checkpoint().unwrap(),
            &expected_state,
            expected_state.layout().len(),
            "transformed versus expanded scalar state",
        );
    }
}
