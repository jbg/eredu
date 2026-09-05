//! End-to-end replicated realtime construction through the neutral reference backend.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    convert::Infallible,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use eredu_architectures::moshi::{
    self, MoshiConfig, MoshiRealtimeArchitectureVisitor, MoshiRealtimeExecutionArchitecture,
    MoshiRealtimeRequest, PreparedMoshiRealtimeArchitecture,
};
use eredu_checkpoint::{
    recipe::RecipeCatalog,
    schema::StoredDtypeConstraint,
    store::{
        CheckpointLease, CheckpointSource, MemoryWeightStore, PreparedCheckpointSource,
        PreparedTensorSource, SharedCheckpointSource, StoreError, TensorMetadata,
        TensorReadRequest, TensorSelection, WeightStoreBackend, WeightStoreDiagnostics,
    },
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
    StoredDtype,
};
use eredu_core::{
    artifact::{fingerprint_filesystem_artifact, ArtifactFile},
    scheduler::{RequestId, RequestStatus, SchedulerLimits, SemanticStateTransaction},
    Completion, CompletionCancellationMode, ParallelRankTopology, ParallelTopology,
    QuantizationRequest, RealtimeInputFrame, RealtimeSampling, SessionCapabilities, TokenFilter,
    MODEL_LOGITS_OBSERVATION_PATH,
};
use eredu_nn::{
    AttentionCache, AttentionMask, AttentionRequest, EmbeddingLookupPolicy, EmbeddingOperator,
    EmbeddingSpec, Error, GroupSelection, GroupSelectionOperator, GroupedGatedProductOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, GroupedRelu2Operator, GroupedRelu2Spec, Index,
    LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, PadMode, ParameterMetadata, ParameterVisitor,
    ParameterVisitorMut, Parameterized, RotaryOperator, RotaryPosition, RotarySpec, Tensor,
    TensorParallelGroupedGatedProductOperator, TensorParallelGroupedOutput,
    TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec, VocabularyParallelRange,
};
use eredu_runtime::{
    bind_materialized_unit, construct_realtime_model, execute_realtime_frame, materialize_bindings,
    ActivationObserver, CacheResidencyPolicy, CommunicationCompletionCapabilities,
    CommunicationCompletionPolicy, CompletedRealtimeFrame, DeviceState, ExecutionResidency,
    LayerWeightResidency, LayeredArchitecture, MaterializedRealtimeInput, MaterializedUnit,
    ParameterBackend, ParameterGroupOwner, PenaltyConfig, PipelineActivationDtype,
    RealizedRealtimePolicy, RealizedRealtimeState, RealtimeArchitectureIdentity,
    RealtimeCompletionCreationError, RealtimeFrameCompletionMechanism,
    RealtimeFrameTensorMechanisms, RealtimeGenerationState, RealtimeHostTokenMaterializer,
    RealtimeIdentity, RealtimeLayerwiseRuntime, RealtimeMaterializationTask, RealtimeMechanism,
    RealtimeMechanismCapabilities, RealtimeModelConstructionMechanisms,
    RealtimeModelSessionIdentity, RealtimeObservationRequirements, RealtimePayloadContract,
    RealtimePayloadGeneration, RealtimePayloadHistory, RealtimePayloadOwnerIdentity,
    RealtimePayloadState, RealtimeSessionScheduler, ResettableRuntimeLayerState,
    ResidentUnitWindow, RuntimeLayerState, RuntimeState, RuntimeStateComponents, Sampler,
    SamplingBackend, StateComponentMechanism, StateComponentPlacement, StateError,
    StateMechanismCapabilities, StaticParameterVisitorMut, SubmissionBackend,
    SubmittedRealtimeFrame, TokenDomain, WeightBinding, WeightLoweringCapability,
    WeightLoweringKind,
};

include!("support/reference_backend.rs");

/// Exact header catalog whose payload path is deliberately unavailable.
///
/// The reference backend materializes architecture recipes from metadata, so
/// this keeps the production proof practical even for the released 7B
/// PersonaPlex geometry while making accidental payload access fail loudly.
struct MetadataCheckpointSource {
    tensors: BTreeMap<String, TensorMetadata>,
}

impl MetadataCheckpointSource {
    fn from_config(config: &MoshiConfig) -> Self {
        let plan = moshi::safetensors_plan(config).expect("valid reference checkpoint plan");
        let tensors = plan
            .common_tensors
            .iter()
            .map(|tensor| {
                let stored_dtype = match &tensor.dtype {
                    StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                    StoredDtypeConstraint::Floating => StoredDtype::F32,
                    StoredDtypeConstraint::OneOf(dtypes) => dtypes[0].clone(),
                };
                let element_bytes = match stored_dtype {
                    StoredDtype::F32 | StoredDtype::I32 | StoredDtype::U32 => 4,
                    StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::I16 | StoredDtype::U16 => 2,
                    _ => 1,
                };
                let encoded_byte_len = tensor
                    .shape
                    .iter()
                    .try_fold(element_bytes, |bytes: u64, dimension| {
                        bytes.checked_mul(u64::try_from(*dimension).ok()?)
                    })
                    .expect("released reference tensor size fits u64");
                (
                    tensor.key.clone(),
                    TensorMetadata {
                        name: tensor.key.clone(),
                        logical_shape: tensor.shape.clone(),
                        physical_shape: tensor.shape.clone(),
                        stored_dtype,
                        encoded_byte_len,
                        backing_shard: Some("metadata-only.safetensors".into()),
                    },
                )
            })
            .collect();
        Self { tensors }
    }

    fn with_causal_value(mut self, parameter: impl Into<String>, value: i32) -> Self {
        self.tensors
            .get_mut(&parameter.into())
            .expect("causal reference parameter exists")
            .backing_shard = Some(format!("causal:{value}").into());
        self
    }
}

impl SafetensorsCatalog for MetadataCheckpointSource {
    fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.tensors
            .get(key)
            .map(|metadata| CatalogTensorMetadata {
                shape: metadata.logical_shape.clone(),
                stored_dtype: metadata.stored_dtype.clone(),
            })
            .ok_or_else(|| format!("unknown reference tensor {key:?}"))
    }
}

impl RecipeCatalog for MetadataCheckpointSource {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source_metadata(key)
    }
}

impl CheckpointSource for MetadataCheckpointSource {
    fn source_keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        Err(StoreError::Internal(format!(
            "reference proof unexpectedly opened payload {:?}",
            request.key
        )))
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        Ok(WeightStoreDiagnostics {
            backend: WeightStoreBackend::Memory,
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,
            currently_cached_shards: 0,
            touched_shard_paths: Vec::new(),
            payload_shard_paths: Vec::new(),
            physical_reads: 0,
            physical_read_bytes: 0,
            coalesced_group_hits: 0,
        })
    }
}

impl Clone for ReferenceCompletion {
    fn clone(&self) -> Self {
        Self
    }
}

type ReferenceState = DeviceState<ReferenceBackend, ReferenceCache>;
type ReferenceGeneration = RealtimeGenerationState<
    RealtimePayloadState<TransactionalReferenceState, ReferenceTensor>,
    ReferenceSampler,
    i32,
    ReferenceCompletion,
>;

#[derive(Clone)]
struct TransactionalReferenceState(ReferenceState);

impl TransactionalReferenceState {
    fn attention_offset(&self) -> i32 {
        self.0.as_ref().iter().map(|cache| cache.offset).sum()
    }
}

#[derive(Clone)]
struct TransactionalReferenceBranch(ReferenceState);

impl Deref for TransactionalReferenceBranch {
    type Target = ReferenceState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TransactionalReferenceBranch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl SemanticStateTransaction for TransactionalReferenceState {
    type Branch = TransactionalReferenceBranch;
    type Error = Infallible;

    fn branch(&self) -> Result<Self::Branch, Self::Error> {
        Ok(TransactionalReferenceBranch(self.0.clone()))
    }

    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error> {
        self.0 = branch.0;
        Ok(())
    }

    fn discard_branch(_branch: Self::Branch) -> Result<(), Self::Error> {
        Ok(())
    }

    fn permits_parallel_branches(&self) -> bool {
        true
    }
}

#[derive(Default)]
struct ReferenceFrameTensorMechanisms;

impl RealtimeHostTokenMaterializer for ReferenceFrameTensorMechanisms {
    type Tensor = ReferenceTensor;
    type Error = Infallible;

    fn materialize_i32(
        &mut self,
        _values: &[i32],
        shape: [usize; 2],
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(
            shape
                .into_iter()
                .map(|dimension| i32::try_from(dimension).expect("tiny frame dimensions fit i32"))
                .collect(),
        ))
    }
}

impl RealtimeFrameTensorMechanisms for ReferenceFrameTensorMechanisms {
    type Tensor = ReferenceTensor;
    type Error = Infallible;

    fn column(
        &mut self,
        matrix: &Self::Tensor,
        _column: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![matrix.0[0], 1]))
    }

    fn filled_column(&mut self, _token: i32, batch: usize) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            i32::try_from(batch).expect("tiny frame batch fits i32"),
            1,
        ]))
    }

    fn stack_columns(
        &mut self,
        columns: &[Self::Tensor],
        batch: usize,
    ) -> Result<Self::Tensor, Self::Error> {
        Ok(ReferenceTensor(vec![
            i32::try_from(batch).expect("tiny frame batch fits i32"),
            i32::try_from(columns.len()).expect("tiny frame columns fit i32"),
        ]))
    }
}

#[derive(Default)]
struct ReferenceFrameCompletionMechanism {
    calls: usize,
    retained_calls: usize,
    expected_text_cardinality: i32,
    expected_text_frames: i32,
}

struct ReferenceRealtimeObserver {
    paths: Rc<RefCell<Vec<String>>>,
    values: Rc<RefCell<Vec<ReferenceTensor>>>,
    replacements: BTreeMap<String, ReferenceTensor>,
    fail_on: Option<String>,
}

impl ActivationObserver<ReferenceTensor, Error> for ReferenceRealtimeObserver {
    fn observe(&mut self, path: &str, value: &ReferenceTensor) -> Result<(), Error> {
        self.paths.borrow_mut().push(path.to_owned());
        self.values.borrow_mut().push(value.clone());
        if self.fail_on.as_deref() == Some(path) {
            return Err(Error::backend(format!(
                "reference observer rejected {path}"
            )));
        }
        Ok(())
    }

    fn intervene(
        &mut self,
        path: &str,
        _value: &ReferenceTensor,
    ) -> Result<Option<ReferenceTensor>, Error> {
        Ok(self.replacements.get(path).cloned())
    }
}

impl<T>
    RealtimeFrameCompletionMechanism<
        ReferenceTensor,
        T,
        (ReferenceTensor, moshi::ForwardContext<ReferenceTensor>),
    > for ReferenceFrameCompletionMechanism
{
    type Completion = ReferenceCompletion;
    type Error = Infallible;

    fn complete(
        &mut self,
        _input: MaterializedRealtimeInput<ReferenceTensor>,
        output: &CompletedRealtimeFrame<ReferenceTensor, ReferenceTensor>,
        _model_state: &T,
        _payload_history: &RealtimePayloadHistory<ReferenceTensor>,
        execution: Option<(ReferenceTensor, moshi::ForwardContext<ReferenceTensor>)>,
    ) -> Result<Self::Completion, RealtimeCompletionCreationError<Self::Completion, Self::Error>>
    {
        self.calls += 1;
        let completion = if let Some((text_logits, forward)) = execution {
            self.retained_calls += 1;
            assert_eq!(
                text_logits,
                ReferenceTensor(vec![
                    1,
                    self.expected_text_frames.max(1),
                    self.expected_text_cardinality,
                ])
            );
            assert_eq!(forward.text_logits(), Some(&text_logits));
            ReferenceBackend::submit(&(), [output.text(), output.sampled_audio(), &text_logits])
                .expect("reference submission is infallible")
        } else {
            ReferenceBackend::submit(&(), [output.text(), output.sampled_audio()])
                .expect("reference submission is infallible")
        };
        Ok(completion)
    }
}

#[derive(Debug, Default)]
struct ConstructionTrace {
    materialization_calls: usize,
    materialization_tasks: usize,
    resident_policy_calls: usize,
    resident_units: usize,
    bounded_policy_calls: usize,
    generated_components: usize,
    transforming_tasks: usize,
    source_architectures: usize,
    state_calls: usize,
}

struct ReferenceConstructionMechanisms {
    trace: Rc<RefCell<ConstructionTrace>>,
    store: SharedCheckpointSource,
    independent_authority: Option<Rc<IndependentAuthorityResources>>,
}

struct StaticMaterializationBinder<'a> {
    units: &'a mut BTreeMap<ParameterGroupOwner, MaterializedUnit<ReferenceBackend>>,
}

impl StaticParameterVisitorMut<ReferenceBackend> for StaticMaterializationBinder<'_> {
    type Error = String;

    fn visit_mut<M>(&mut self, role: &str, module: &mut M) -> Result<(), Self::Error>
    where
        M: Parameterized<ReferenceTensor>,
    {
        let owner = ParameterGroupOwner::static_role(role);
        let materialized = self
            .units
            .remove(&owner)
            .ok_or_else(|| format!("missing materialized static role {role:?}"))?;
        bind_materialized_unit::<ReferenceBackend, _>(module, materialized)
            .map_err(|error| error.to_string())
    }
}

impl<A> RealtimeModelConstructionMechanisms<A, ReferenceBackend> for ReferenceConstructionMechanisms
where
    A: LayeredArchitecture<ReferenceBackend, ReferenceState>,
    A::Error: std::fmt::Display,
{
    type State = ReferenceState;
    type PolicyError = eredu_runtime::ResidentUnitWindowError;
    type ResidentPolicy = ResidentUnitWindow<A::Unit>;
    type BoundedPolicy = ResidentUnitWindow<A::Unit>;
    type Error = String;

    fn prepare_resident_materialization(
        &mut self,
        architecture: &mut A,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[RealtimeMaterializationTask],
        selected: &eredu_runtime::SelectedRealtimeRealization,
        _context: &(),
    ) -> Result<(), Self::Error> {
        if let Some(authority) = &self.independent_authority {
            let selected_elements = match &authority.placement {
                TensorSelection::Range { start, end, .. } => end - start,
                TensorSelection::Indices { indices, .. } => indices.len(),
                TensorSelection::Contiguous { shape, .. } => shape.iter().product(),
                TensorSelection::Full => authority
                    .bound_module
                    .weight
                    .0
                    .iter()
                    .map(|dimension| usize::try_from(*dimension).unwrap())
                    .product(),
            };
            assert_eq!(
                authority.bound_module.weight,
                ReferenceTensor(vec![i32::try_from(selected_elements).unwrap()])
            );
            authority
                .construction_uses
                .set(authority.construction_uses.get() + 1);
        }
        let transforms = tasks
            .iter()
            .filter(|task| {
                matches!(
                    task.lowering().kind(),
                    WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
                )
            })
            .count();
        assert_eq!(source_architecture.is_some(), transforms > 0);
        assert!(!source_units.is_some() || transforms > 0);
        assert_eq!(tasks.len(), selected.weight_lowerings().len());
        let mut bindings = BTreeMap::<ParameterGroupOwner, Vec<WeightBinding>>::new();
        let mut generated_components = 0;
        for task in tasks {
            let owner_bindings = bindings.entry(task.owner().clone()).or_default();
            for component in task.components() {
                let Some((recipe, output)) = component.recipe().zip(component.recipe_output())
                else {
                    generated_components += 1;
                    continue;
                };
                owner_bindings.push(
                    WeightBinding::from_recipe(
                        component.requirement().target().as_str(),
                        recipe.clone(),
                        output.byte_len(),
                    )
                    .map_err(|error| error.to_string())?,
                );
            }
        }
        let mut materialized = bindings
            .into_iter()
            .map(|(owner, bindings)| {
                materialize_bindings::<ReferenceBackend>(self.store.as_ref(), &bindings, &())
                    .map(|unit| (owner, unit))
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        architecture
            .visit_static_parameters_mut(&mut StaticMaterializationBinder {
                units: &mut materialized,
            })
            .map_err(|error| error.to_string())?;
        for (ordinal, unit) in units.iter_mut().enumerate() {
            let address = selected
                .execution_units()
                .address(ordinal)
                .ok_or_else(|| format!("missing execution address for unit {ordinal}"))?;
            let group = selected
                .execution_units()
                .group_id(address.group())
                .ok_or_else(|| format!("missing execution group for unit {ordinal}"))?
                .clone();
            let owner = ParameterGroupOwner::execution_unit(group, address.index());
            let materialized_unit = materialized
                .remove(&owner)
                .ok_or_else(|| format!("missing materialized execution unit {owner:?}"))?;
            bind_materialized_unit::<ReferenceBackend, _>(unit, materialized_unit)
                .map_err(|error| error.to_string())?;
        }
        if !materialized.is_empty() {
            return Err(format!(
                "unbound materialized owners: {:?}",
                materialized.keys().collect::<Vec<_>>()
            ));
        }
        let mut trace = self.trace.borrow_mut();
        trace.materialization_calls += 1;
        trace.materialization_tasks += tasks.len();
        trace.resident_units = units.len();
        trace.generated_components += generated_components;
        trace.transforming_tasks += transforms;
        trace.source_architectures += usize::from(source_architecture.is_some());
        Ok(())
    }

    fn resident_policy(
        &mut self,
        _architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &eredu_runtime::SelectedRealtimeRealization,
        _context: &(),
    ) -> Result<RealizedRealtimePolicy<Self::ResidentPolicy>, Self::Error> {
        assert_eq!(units.len(), selected.execution_units().len());
        self.trace.borrow_mut().resident_policy_calls += 1;
        Ok(RealizedRealtimePolicy::new(
            ResidentUnitWindow::new(units),
            selected.residency(),
        ))
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        source_architecture: Option<&mut A>,
        tasks: &[RealtimeMaterializationTask],
        selected: &eredu_runtime::SelectedRealtimeRealization,
        context: &(),
    ) -> Result<RealizedRealtimePolicy<Self::BoundedPolicy>, Self::Error> {
        let mut units = (0..selected.execution_units().len())
            .map(|ordinal| {
                let address = selected
                    .execution_units()
                    .address(ordinal)
                    .ok_or_else(|| format!("missing bounded execution address {ordinal}"))?;
                architecture
                    .build_unit(address.group(), address.index(), context)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        // This reference policy retains its structural units, but it is entered
        // exclusively through the selected bounded constructor branch.
        self.prepare_resident_materialization(
            architecture,
            &mut units,
            source_architecture,
            None,
            tasks,
            selected,
            context,
        )?;
        self.trace.borrow_mut().bounded_policy_calls += 1;
        Ok(RealizedRealtimePolicy::new(
            ResidentUnitWindow::new(units),
            selected.residency(),
        ))
    }

    fn realize_state(
        &mut self,
        selected: &eredu_runtime::SelectedRealtimeStateRealization,
        _context: &(),
    ) -> Result<RealizedRealtimeState<Self::State>, Self::Error> {
        self.trace.borrow_mut().state_calls += 1;
        let state = DeviceState::create(selected.layout().clone(), |_, policy| {
            Ok::<_, std::convert::Infallible>(ReferenceCache {
                offset: 0,
                window: policy
                    .attention()
                    .and_then(|attention| attention.window())
                    .map(|window| window.get() as i32),
                resets: 0,
                fixed: None,
            })
        })
        .map_err(|error| error.to_string())?;
        Ok(RealizedRealtimeState::new(state, selected.clone()))
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ConstructionSummary {
    task_count: usize,
    unit_count: usize,
    state_layers: usize,
    committed_attention_offset: i32,
    submitted_frames: usize,
    executed_frames: usize,
    random_after_sampled: i32,
    random_after_forced: i32,
    resumed_frontier: usize,
}

struct ConstructionVisitor {
    started: Rc<Cell<bool>>,
    selection_complete: Rc<Cell<bool>>,
    ingress: eredu_runtime::RealtimeIngressContract,
    residency: LayerWeightResidency,
    expects_transform: bool,
    text_cardinality: i32,
    inspection_authority: Option<IndependentSessionAuthority>,
}

#[derive(Clone)]
struct IndependentBoundModule {
    weight: ReferenceTensor,
}

fn independent_parameter_metadata() -> ParameterMetadata {
    ParameterMetadata {
        id: eredu_nn::ParameterId::new("weight").unwrap(),
        trainable: false,
        alias_of: None,
        group: None,
        linear_companion: None,
        linear_companion_of: None,
    }
}

impl Parameterized<ReferenceTensor> for IndependentBoundModule {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, ReferenceTensor>,
    {
        visitor.visit(independent_parameter_metadata(), &self.weight);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, ReferenceTensor>,
    {
        visitor.visit_mut(independent_parameter_metadata(), &mut self.weight);
    }

    fn set_trainable(&mut self, _trainable: bool) {}
}

struct IndependentAuthorityResources {
    bound_module: IndependentBoundModule,
    placement: TensorSelection,
    communication: eredu_runtime::PreparedCommunicationRealization,
    construction_uses: Cell<usize>,
    group_callbacks: Cell<usize>,
    route_callbacks: Cell<usize>,
    bound_value_trace: RefCell<Vec<i32>>,
    causal_value: i32,
    executed_outputs: RefCell<Vec<(ReferenceTensor, ReferenceTensor)>>,
    group_resources: RefCell<Vec<MockCommunicationResource>>,
    route_resources: RefCell<Vec<MockCommunicationResource>>,
    omit_route_resources: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct MockCommunicationResource {
    ordinal: usize,
    world_wave: bool,
}

struct IndependentSessionAuthority {
    inspection: eredu_core::ArtifactInspection<()>,
    admission: eredu_core::PreparationAdmission,
    admitted_report: eredu_core::ModelInspectionReport,
    resources: Rc<IndependentAuthorityResources>,
    realized: Rc<RefCell<Option<IndependentRealizedSession>>>,
}

#[derive(Debug)]
struct IndependentRealizedSession {
    capabilities: SessionCapabilities,
    report: eredu_core::ModelInspectionReport,
    scheduler_capabilities: eredu_core::scheduler::SchedulerCapabilities,
    scheduler_report: eredu_core::scheduler::SchedulerReport,
    executed_outputs: Vec<(ReferenceTensor, ReferenceTensor)>,
}

impl MoshiRealtimeArchitectureVisitor<ReferenceBackend, ReferenceState> for ConstructionVisitor {
    type Output = ConstructionSummary;
    type Error = String;

    fn construction_started(&mut self) {
        assert!(self.selection_complete.get());
        assert!(!self.started.replace(true));
    }

    fn visit<A>(
        self,
        mut prepared: PreparedMoshiRealtimeArchitecture<A>,
        store: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: MoshiRealtimeExecutionArchitecture<ReferenceBackend, ReferenceState>
            + RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        assert!(self.started.get());
        let independent_authority = self
            .inspection_authority
            .as_ref()
            .map(|authority| authority.resources.clone());
        let architecture = prepared.take_architecture();
        let source_architecture = prepared.take_source_architecture();
        assert_eq!(source_architecture.is_some(), self.expects_transform);
        let contract = prepared.take_contract();
        let task_count = contract.tasks().len();
        let expected_units = contract.selected().execution_units().len();
        let expected_state_layers = contract.selected().state().layout().len();
        let model_identity = RealtimeModelSessionIdentity::from_selected(contract.selected());
        assert_eq!(contract.selected().residency(), self.residency);
        let trace = Rc::new(RefCell::new(ConstructionTrace::default()));
        let constructed = construct_realtime_model::<A, ReferenceBackend, _>(
            architecture,
            source_architecture,
            contract,
            ReferenceConstructionMechanisms {
                trace: trace.clone(),
                store,
                independent_authority,
            },
            &(),
        )
        .map_err(|error| error.to_string())?;

        assert_eq!(
            matches!(
                constructed.execution(),
                RealtimeLayerwiseRuntime::Resident(_)
            ),
            matches!(self.residency, LayerWeightResidency::FullyResident)
        );
        assert_eq!(constructed.state().layout().len(), expected_state_layers);
        let construction = trace.borrow();
        assert_eq!(construction.materialization_calls, 1);
        assert_eq!(construction.materialization_tasks, task_count);
        assert_eq!(construction.resident_units, expected_units);
        assert_eq!(construction.state_calls, 1);
        assert_eq!(
            construction.source_architectures,
            usize::from(self.expects_transform)
        );
        assert_eq!(construction.transforming_tasks > 0, self.expects_transform);
        assert_eq!(
            construction.resident_policy_calls,
            usize::from(matches!(
                self.residency,
                LayerWeightResidency::FullyResident
            ))
        );
        assert_eq!(
            construction.bounded_policy_calls,
            usize::from(!matches!(
                self.residency,
                LayerWeightResidency::FullyResident
            ))
        );
        drop(construction);
        let (execution, state) = constructed.into_execution_and_state();
        let schedule = self.ingress.schedule().clone();
        let payload_state = RealtimePayloadState::new(
            TransactionalReferenceState(state),
            RealtimePayloadHistory::new(schedule.clone()),
            &schedule,
        )
        .map_err(|error| error.to_string())?;
        let sampling = RealtimeSampling::new(1.0, 1.0, 7).map_err(|error| error.to_string())?;
        let generation =
            RealtimeGenerationState::<_, ReferenceSampler, i32, ReferenceCompletion>::new(
                payload_state,
                schedule.clone(),
                sampling,
                vec![ReferenceSampler; schedule.depth_audio_codebooks() + 1],
                Some(i32::try_from(sampling.seed()).expect("reference seed fits i32")),
            )
            .map_err(|error| error.to_string())?;
        let initial_offset = generation.model_state().model_state().attention_offset();
        let mut executor = moshi::MoshiPreparedRealtimeFrameExecutor::new(execution);
        let mut host = ReferenceFrameTensorMechanisms;
        let mut tensors = ReferenceFrameTensorMechanisms;
        let authority_value = self
            .inspection_authority
            .as_ref()
            .map_or(1, |authority| authority.resources.bound_module.weight.0[0]);
        let causal_step = self
            .inspection_authority
            .as_ref()
            .map_or(1, |authority| authority.resources.causal_value);
        let mut completion = ReferenceFrameCompletionMechanism {
            expected_text_cardinality: self.text_cardinality,
            expected_text_frames: authority_value,
            ..ReferenceFrameCompletionMechanism::default()
        };
        type ReferenceOutput = SubmittedRealtimeFrame<ReferenceTensor, ReferenceCompletion>;
        type ReferenceSessions = RealtimeSessionScheduler<
            RealtimePayloadState<TransactionalReferenceState, ReferenceTensor>,
            ReferenceSampler,
            i32,
            ReferenceCompletion,
            ReferenceOutput,
        >;
        let mut sessions = ReferenceSessions::new(
            model_identity,
            SchedulerLimits::with_execution_bounds(1, 4, 1, 1, 1, usize::MAX)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let request = RequestId::new(7);
        sessions
            .register(request, generation)
            .map_err(|error| error.to_string())?;
        sessions
            .enqueue_batch(
                request,
                (0..2)
                    .map(|_| {
                        RealtimeInputFrame::new(1, vec![1; schedule.input_audio_codebooks()])
                            .with_forced_text(vec![authority_value])
                            .with_diagnostics()
                    })
                    .collect(),
            )
            .map_err(|error| error.to_string())?;
        let observed_frames = Rc::new(Cell::new(0));
        clear_reference_trace();
        for _ in 0..2 {
            if observed_frames.get() == 0 {
                if let Some(authority) = &self.inspection_authority {
                    let groups = authority.resources.communication.try_create_groups(
                        |descriptor, world_wave| {
                            authority
                                .resources
                                .group_callbacks
                                .set(authority.resources.group_callbacks.get() + 1);
                            Ok::<_, String>(MockCommunicationResource {
                                ordinal: descriptor.creation_order(),
                                world_wave,
                            })
                        },
                    )?;
                    let mut routes = authority.resources.communication.try_create_routes(
                        |descriptor, world_wave| {
                            authority
                                .resources
                                .route_callbacks
                                .set(authority.resources.route_callbacks.get() + 1);
                            Ok::<_, String>(MockCommunicationResource {
                                ordinal: descriptor.submission_order(),
                                world_wave,
                            })
                        },
                    )?;
                    if authority.resources.omit_route_resources {
                        routes.clear();
                    }
                    *authority.resources.group_resources.borrow_mut() = groups;
                    *authority.resources.route_resources.borrow_mut() = routes;
                }
            }
            if let Some(authority) = &self.inspection_authority {
                if authority.resources.group_resources.borrow().len()
                    != authority.resources.communication.manifest().groups().len()
                    || authority.resources.route_resources.borrow().len()
                        != authority.resources.communication.manifest().routes().len()
                {
                    return Err("prepared communication resources are incomplete".into());
                }
                authority
                    .resources
                    .bound_value_trace
                    .borrow_mut()
                    .push(authority_value);
            }
            let observed_frames = observed_frames.clone();
            sessions
                .run_local_turn(Instant::now(), |_, frame, branch| {
                    let payload_contract = branch
                        .payload_contract(&self.ingress)
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    let output =
                        execute_realtime_frame::<ReferenceBackend, _, _, _, _, _, _, _, _>(
                            &self.ingress,
                            &payload_contract,
                            frame,
                            branch.generation_mut(),
                            &eredu_architectures::moshi::realtime_decision_execution(),
                            &mut host,
                            &mut tensors,
                            &mut executor,
                            &mut completion,
                            &(),
                        )
                        .map_err(|error| std::io::Error::other(error.to_string()))?;
                    if let Some(authority) = &self.inspection_authority {
                        authority.resources.executed_outputs.borrow_mut().push((
                            output.frame().text().clone(),
                            output.frame().sampled_audio().clone(),
                        ));
                    }
                    if !output.frame().diagnostics().is_empty() {
                        assert_eq!(output.frame().text(), &ReferenceTensor(vec![1, 1]));
                        assert_eq!(
                            output.frame().sampled_audio(),
                            &ReferenceTensor(vec![
                                1,
                                i32::try_from(schedule.generated_audio_codebooks()).unwrap(),
                            ])
                        );
                        assert_eq!(
                            output.frame().diagnostics().len(),
                            schedule.depth_audio_codebooks() + 1
                        );
                    }
                    observed_frames.set(observed_frames.get() + 1);
                    Ok::<ReferenceOutput, std::io::Error>(output)
                })
                .map_err(|error| error.to_string())?;
        }
        assert_eq!(observed_frames.get(), 2);
        assert_eq!(completion.calls, 2);
        assert!(completion.retained_calls >= 1);
        assert!(!reference_trace().linear_outputs.is_empty());
        let generation = sessions.request_state(request).unwrap().generation();
        let committed_attention_offset = generation.model_state().model_state().attention_offset();
        assert!(committed_attention_offset > initial_offset);
        assert_eq!(generation.schedule_state().frontier(), 2);
        let random_after_sampled = *generation.random_state().unwrap();
        assert_eq!(
            random_after_sampled,
            7 + causal_step * i32::try_from(completion.retained_calls).unwrap()
        );

        let incarnation = sessions.request_state(request).unwrap().incarnation();
        let released = sessions
            .release(request)
            .map_err(|error| error.to_string())?;
        assert!(sessions.request_state(request).is_none());
        let resumed_request = RequestId::new(8);
        sessions
            .resume(resumed_request, released)
            .map_err(|error| error.to_string())?;
        assert_eq!(
            sessions
                .request_state(resumed_request)
                .unwrap()
                .incarnation(),
            incarnation
        );
        sessions
            .enqueue(
                resumed_request,
                RealtimeInputFrame::new(1, vec![1; schedule.input_audio_codebooks()])
                    .with_forced_text(vec![authority_value])
                    .with_forced_generated_audio(vec![1; schedule.generated_audio_codebooks()])
                    .with_diagnostics(),
            )
            .map_err(|error| error.to_string())?;
        sessions
            .run_local_turn(Instant::now(), |_, frame, branch| {
                let payload_contract = branch
                    .payload_contract(&self.ingress)
                    .map_err(|error| std::io::Error::other(error.to_string()))?;
                execute_realtime_frame::<ReferenceBackend, _, _, _, _, _, _, _, _>(
                    &self.ingress,
                    &payload_contract,
                    frame,
                    branch.generation_mut(),
                    &eredu_architectures::moshi::realtime_decision_execution(),
                    &mut host,
                    &mut tensors,
                    &mut executor,
                    &mut completion,
                    &(),
                )
                .map_err(|error| std::io::Error::other(error.to_string()))
            })
            .map_err(|error| error.to_string())?;
        let resumed = sessions
            .request_state(resumed_request)
            .unwrap()
            .generation();
        let random_after_forced = *resumed.random_state().unwrap();
        let unforced_depth_tail = schedule
            .depth_audio_codebooks()
            .saturating_sub(schedule.generated_audio_codebooks())
            .saturating_sub(1);
        assert_eq!(
            random_after_forced,
            random_after_sampled + causal_step * i32::try_from(unforced_depth_tail).unwrap()
        );
        assert_eq!(resumed.schedule_state().frontier(), 3);
        if let Some(authority) = self.inspection_authority {
            let realized_capabilities = SessionCapabilities::new(
                sessions.request_state(resumed_request).is_some(),
                completion.calls > 0,
                observed_frames.get() > 0 && !reference_trace().linear_outputs.is_empty(),
            );
            assert_eq!(
                realized_capabilities,
                authority.admission.session_capabilities()
            );
            let resources = &authority.resources;
            let realized_report = eredu_core::finalize_realized_model_inspection(
                &authority.admitted_report,
                eredu_core::RealizedInspectionOutcomes {
                    artifact_authority: authority.inspection.path()
                        == authority.admitted_report.path,
                    binding: resources.construction_uses.get() > 0
                        && resources.bound_module.weight == ReferenceTensor(vec![authority_value]),
                    construction: {
                        let construction = trace.borrow();
                        construction.materialization_calls == 1 && construction.state_calls == 1
                    },
                    communication: resources.group_callbacks.get()
                        == resources.communication.manifest().groups().len()
                        && resources.route_callbacks.get()
                            == resources.communication.manifest().routes().len(),
                    execution: observed_frames.get() > 0
                        && completion.calls > 0
                        && resources
                            .bound_value_trace
                            .borrow()
                            .iter()
                            .all(|value| *value == authority_value),
                    session: realized_capabilities,
                },
            );
            assert_eq!(realized_report, authority.admitted_report);
            *authority.realized.borrow_mut() = Some(IndependentRealizedSession {
                capabilities: realized_capabilities,
                report: realized_report,
                scheduler_capabilities: sessions.capabilities(),
                scheduler_report: sessions.report(),
                executed_outputs: resources.executed_outputs.borrow().clone(),
            });
        }
        Ok(ConstructionSummary {
            task_count,
            unit_count: expected_units,
            state_layers: expected_state_layers,
            committed_attention_offset,
            submitted_frames: completion.calls,
            executed_frames: completion.retained_calls,
            random_after_sampled,
            random_after_forced,
            resumed_frontier: resumed.schedule_state().frontier(),
        })
    }
}

#[derive(Debug)]
struct ObservationSummary {
    paths: Vec<String>,
    observed_values: Vec<ReferenceTensor>,
    sampled_logits: Vec<Vec<i32>>,
    frontier: usize,
    attention_offset: i32,
    random: i32,
    completion_calls: usize,
    request_status: RequestStatus,
}

struct ObservationVisitor {
    ingress: eredu_runtime::RealtimeIngressContract,
    observer: ReferenceRealtimeObserver,
    expected_output_cardinality: i32,
}

impl MoshiRealtimeArchitectureVisitor<ReferenceBackend, ReferenceState> for ObservationVisitor {
    type Output = ObservationSummary;
    type Error = String;

    fn visit<A>(
        self,
        mut prepared: PreparedMoshiRealtimeArchitecture<A>,
        store: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: MoshiRealtimeExecutionArchitecture<ReferenceBackend, ReferenceState>
            + RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        let architecture = prepared.take_architecture();
        let contract = prepared.take_contract();
        let model_identity = RealtimeModelSessionIdentity::from_selected(contract.selected());
        let trace = Rc::new(RefCell::new(ConstructionTrace::default()));
        let constructed = construct_realtime_model::<A, ReferenceBackend, _>(
            architecture,
            None,
            contract,
            ReferenceConstructionMechanisms {
                trace,
                store,
                independent_authority: None,
            },
            &(),
        )
        .map_err(|error| error.to_string())?;
        let (execution, state) = constructed.into_execution_and_state();
        let schedule = self.ingress.schedule().clone();
        let payload_state = RealtimePayloadState::new(
            TransactionalReferenceState(state),
            RealtimePayloadHistory::new(schedule.clone()),
            &schedule,
        )
        .map_err(|error| error.to_string())?;
        let sampling = RealtimeSampling::new(1.0, 1.0, 7).map_err(|error| error.to_string())?;
        let generation =
            RealtimeGenerationState::<_, ReferenceSampler, i32, ReferenceCompletion>::new(
                payload_state,
                schedule.clone(),
                sampling,
                vec![ReferenceSampler; schedule.depth_audio_codebooks() + 1],
                Some(7),
            )
            .map_err(|error| error.to_string())?;
        let initial_offset = generation.model_state().model_state().attention_offset();
        let observation_paths = Rc::clone(&self.observer.paths);
        let observed_values = Rc::clone(&self.observer.values);
        let expects_failure = self.observer.fail_on.is_some();
        let mut executor =
            moshi::MoshiPreparedRealtimeFrameExecutor::with_observer(execution, self.observer);
        let mut host = ReferenceFrameTensorMechanisms;
        let mut tensors = ReferenceFrameTensorMechanisms;
        let mut completion = ReferenceFrameCompletionMechanism {
            expected_text_cardinality: self.expected_output_cardinality,
            expected_text_frames: 3,
            ..ReferenceFrameCompletionMechanism::default()
        };
        if expects_failure {
            let payload_contract = RealtimePayloadContract::new(
                schedule.clone(),
                1,
                self.ingress.text_domain(),
                self.ingress.audio_domain(),
                RealtimePayloadGeneration::new(1).unwrap(),
                RealtimePayloadOwnerIdentity::new(1).unwrap(),
            )
            .map_err(|error| error.to_string())?;
            let frame = RealtimeInputFrame::new(1, vec![1; schedule.input_audio_codebooks()])
                .with_diagnostics();
            let mut branch = generation.branch().map_err(|error| error.to_string())?;
            clear_reference_trace();
            let result = execute_realtime_frame::<ReferenceBackend, _, _, _, _, _, _, _, _>(
                &self.ingress,
                &payload_contract,
                &frame,
                &mut branch,
                &eredu_architectures::moshi::realtime_decision_execution(),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            );
            assert!(result.is_err());
            <ReferenceGeneration as SemanticStateTransaction>::discard_branch(branch)
                .map_err(|error| error.to_string())?;
            let attention_offset = generation.model_state().model_state().attention_offset();
            assert_eq!(attention_offset, initial_offset);
            let paths = observation_paths.borrow().clone();
            let observed_values = observed_values.borrow().clone();
            return Ok(ObservationSummary {
                paths,
                observed_values,
                sampled_logits: reference_trace().sampled_logits,
                frontier: generation.schedule_state().frontier(),
                attention_offset,
                random: *generation.random_state().unwrap(),
                completion_calls: completion.calls,
                request_status: RequestStatus::Failed,
            });
        }
        type ReferenceOutput = SubmittedRealtimeFrame<ReferenceTensor, ReferenceCompletion>;
        type ReferenceSessions = RealtimeSessionScheduler<
            RealtimePayloadState<TransactionalReferenceState, ReferenceTensor>,
            ReferenceSampler,
            i32,
            ReferenceCompletion,
            ReferenceOutput,
        >;
        let mut sessions = ReferenceSessions::new(
            model_identity,
            SchedulerLimits::with_execution_bounds(1, 1, 1, 1, 1, usize::MAX)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let request = RequestId::new(28);
        sessions
            .register(request, generation)
            .map_err(|error| error.to_string())?;
        sessions
            .enqueue(
                request,
                RealtimeInputFrame::new(1, vec![1; schedule.input_audio_codebooks()])
                    .with_diagnostics(),
            )
            .map_err(|error| error.to_string())?;
        clear_reference_trace();
        let _ = sessions.run_local_turn(Instant::now(), |_, frame, branch| {
            let payload_contract = branch
                .payload_contract(&self.ingress)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            execute_realtime_frame::<ReferenceBackend, _, _, _, _, _, _, _, _>(
                &self.ingress,
                &payload_contract,
                frame,
                branch.generation_mut(),
                &eredu_architectures::moshi::realtime_decision_execution(),
                &mut host,
                &mut tensors,
                &mut executor,
                &mut completion,
                &(),
            )
            .map_err(|error| std::io::Error::other(error.to_string()))
        });
        let state = sessions
            .request_state(request)
            .ok_or_else(|| "observation request state disappeared".to_owned())?
            .generation();
        let attention_offset = state.model_state().model_state().attention_offset();
        let frontier = state.schedule_state().frontier();
        let random = *state
            .random_state()
            .ok_or_else(|| "observation request lost random state".to_owned())?;
        let request_status = sessions
            .request_status(request)
            .ok_or_else(|| "observation request status disappeared".to_owned())?;
        if request_status == RequestStatus::Failed {
            assert_eq!(attention_offset, initial_offset);
        }
        let paths = observation_paths.borrow().clone();
        let observed_values = observed_values.borrow().clone();
        Ok(ObservationSummary {
            paths,
            observed_values,
            sampled_logits: reference_trace().sampled_logits,
            frontier,
            attention_offset,
            random,
            completion_calls: completion.calls,
            request_status,
        })
    }
}

fn tiny_config() -> MoshiConfig {
    MoshiConfig::from_json(
        r#"{
            "model_type":"moshi", "dim":4, "text_card":17,
            "n_q":2, "dep_q":1, "generated_audio_codebooks":1, "card":16,
            "num_heads":1, "num_layers":1, "dim_feedforward":6,
            "causal":true, "context":3, "max_period":10000.0,
            "positional_embedding":"rope", "depformer_dim":4,
            "depformer_dim_feedforward":6, "depformer_num_heads":1,
            "depformer_num_layers":1, "depformer_context":2,
            "depformer_max_period":10000.0, "depformer_pos_emb":"none",
            "delays":[0,0,1]
        }"#,
    )
    .unwrap()
}

fn quantizable_native_config() -> MoshiConfig {
    MoshiConfig::from_json(
        r#"{
            "model_type":"moshi", "dim":32, "text_card":32,
            "n_q":2, "dep_q":1, "generated_audio_codebooks":1, "card":32,
            "num_heads":1, "num_layers":1, "dim_feedforward":48,
            "causal":true, "context":3, "max_period":10000.0,
            "positional_embedding":"rope", "depformer_dim":32,
            "depformer_dim_feedforward":48, "depformer_num_heads":1,
            "depformer_num_layers":1, "depformer_context":2,
            "depformer_max_period":10000.0, "depformer_pos_emb":"none",
            "delays":[0,0,1]
        }"#,
    )
    .unwrap()
}

fn request(
    quantization: Option<QuantizationRequest>,
    residency: LayerWeightResidency,
) -> MoshiRealtimeRequest {
    MoshiRealtimeRequest::new(
        quantization,
        residency,
        CacheResidencyPolicy::Device,
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        PipelineActivationDtype::Float32,
        CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
        RealtimeObservationRequirements::new(true, []),
    )
}

fn tensor_parallel_request(residency: LayerWeightResidency) -> MoshiRealtimeRequest {
    MoshiRealtimeRequest::new(
        None,
        residency,
        CacheResidencyPolicy::Device,
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 0).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        PipelineActivationDtype::Float32,
        CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
        RealtimeObservationRequirements::new(true, []),
    )
}

fn capabilities(
    requirements: &eredu_runtime::RealtimeArchitectureRequirements,
) -> RealtimeMechanismCapabilities {
    let lowerings = requirements.executions()[0]
        .weight_lowerings()
        .iter()
        .map(|lowering| {
            WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind())
        })
        .collect();
    let layout = requirements.state_layout();
    let state = StateMechanismCapabilities::new((0..layout.len()).flat_map(|layer| {
        layout
            .components(layer)
            .unwrap()
            .iter()
            .cloned()
            .map(move |component| {
                StateComponentMechanism::new(
                    layer,
                    component,
                    Some(StateComponentPlacement::Device),
                    None,
                )
            })
            .collect::<Vec<_>>()
    }))
    .with_transactions(true, true)
    .with_reset(true)
    .with_observation_retention(true);
    RealtimeMechanismCapabilities::new(
        requirements.operators(),
        [
            RealtimeMechanism::TensorOperations,
            RealtimeMechanism::NeuralOperations,
            RealtimeMechanism::ParameterMaterialization,
            RealtimeMechanism::ParameterStorage,
            RealtimeMechanism::StateStorage,
            RealtimeMechanism::CoordinateStorage,
            RealtimeMechanism::Sampling,
            RealtimeMechanism::Randomness,
            RealtimeMechanism::HostConversion,
            RealtimeMechanism::ExactCompletion,
            RealtimeMechanism::ResourceRetention,
            RealtimeMechanism::Transfer,
            RealtimeMechanism::Observation,
            RealtimeMechanism::Collectives,
        ],
        [
            ExecutionResidency::FullyResident,
            ExecutionResidency::LayerwiseHost,
            ExecutionResidency::DenseDiskStream,
        ],
        lowerings,
        state,
        NonZeroUsize::new(2).unwrap(),
        CommunicationCompletionCapabilities::new([
            CompletionCancellationMode::QuarantineUntilComplete,
        ])
        .unwrap(),
        SessionCapabilities::new(true, true, true),
    )
}

fn selected(
    config: &MoshiConfig,
    store: &Arc<MetadataCheckpointSource>,
    request: MoshiRealtimeRequest,
) -> moshi::PreparedMoshiRealtime {
    let preparation = moshi::prepare_realtime_model_from_catalog(
        "reference/header-artifact",
        "reference/header-checkpoint.safetensors",
        config.clone(),
        store.as_ref(),
    )
    .unwrap();
    let inspected = moshi::inspect_moshi_realtime(preparation, request).unwrap();
    let capabilities = capabilities(inspected.requirements()).with_observation_identities(
        moshi::observation_points(config)
            .into_iter()
            .map(|point| RealtimeIdentity::new(point.path()).unwrap()),
    );
    moshi::select_inspected_moshi_realtime(inspected, &capabilities).unwrap()
}

fn run_reference_scenario(
    config: MoshiConfig,
    residency: LayerWeightResidency,
    quantization: Option<QuantizationRequest>,
) -> ConstructionSummary {
    try_run_reference_scenario_with_authority(config, residency, quantization, None).unwrap()
}

fn run_reference_scenario_with_authority(
    config: MoshiConfig,
    residency: LayerWeightResidency,
    quantization: Option<QuantizationRequest>,
    inspection_authority: Option<IndependentSessionAuthority>,
) -> ConstructionSummary {
    try_run_reference_scenario_with_authority(config, residency, quantization, inspection_authority)
        .unwrap()
}

fn try_run_reference_scenario_with_authority(
    config: MoshiConfig,
    residency: LayerWeightResidency,
    quantization: Option<QuantizationRequest>,
    inspection_authority: Option<IndependentSessionAuthority>,
) -> Result<ConstructionSummary, String> {
    let ingress = moshi::realtime_ingress_contract(&config).unwrap();
    let mut store = MetadataCheckpointSource::from_config(&config);
    if let Some(authority) = &inspection_authority {
        store = store.with_causal_value("text_linear.weight", authority.resources.causal_value);
    }
    let store = Arc::new(store);
    let selection_complete = Rc::new(Cell::new(false));
    let started = Rc::new(Cell::new(false));
    let prepared = selected(&config, &store, request(quantization, residency));
    selection_complete.set(true);
    let shared_store: SharedCheckpointSource = store;
    let summary =
        moshi::visit_selected_moshi_realtime_architecture::<ReferenceBackend, ReferenceState, _>(
            prepared,
            shared_store,
            &(),
            ConstructionVisitor {
                started: started.clone(),
                selection_complete,
                ingress,
                residency,
                expects_transform: quantization.is_some(),
                text_cardinality: config.text_vocabulary_size(),
                inspection_authority,
            },
        )
        .map_err(|error| error.to_string())?;
    assert!(started.get());
    assert!(summary.task_count > 0);
    assert!(summary.committed_attention_offset > 0);
    assert_eq!(summary.submitted_frames, 3);
    assert!(summary.executed_frames >= 1);
    assert!(summary.random_after_forced >= summary.random_after_sampled);
    assert_eq!(summary.resumed_frontier, 3);
    Ok(summary)
}

fn run_observation_scenario(
    replacements: BTreeMap<String, ReferenceTensor>,
    fail_on: Option<String>,
    expected_output_cardinality: i32,
) -> ObservationSummary {
    let config = tiny_config();
    let ingress = moshi::realtime_ingress_contract(&config).unwrap();
    let store = Arc::new(MetadataCheckpointSource::from_config(&config));
    let activations = moshi::observation_points(&config)
        .into_iter()
        .map(|point| RealtimeIdentity::new(point.path()).unwrap())
        .collect::<Vec<_>>();
    let request = MoshiRealtimeRequest::new(
        None,
        LayerWeightResidency::FullyResident,
        CacheResidencyPolicy::Device,
        ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        PipelineActivationDtype::Float32,
        CommunicationCompletionPolicy::new(
            Duration::from_secs(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
        RealtimeObservationRequirements::new(true, activations),
    );
    let prepared = selected(&config, &store, request);
    let shared_store: SharedCheckpointSource = store;
    moshi::visit_selected_moshi_realtime_architecture::<ReferenceBackend, ReferenceState, _>(
        prepared,
        shared_store,
        &(),
        ObservationVisitor {
            ingress,
            observer: ReferenceRealtimeObserver {
                paths: Rc::new(RefCell::new(Vec::new())),
                values: Rc::new(RefCell::new(Vec::new())),
                replacements,
                fail_on,
            },
            expected_output_cardinality,
        },
    )
    .unwrap()
}

#[test]
fn declared_realtime_interventions_causally_reach_decisions_and_output_once() {
    let replacements = [
        (
            moshi::ObservationPoint::TemporalInput.path(),
            ReferenceTensor(vec![1, 2, 4]),
        ),
        (
            moshi::ObservationPoint::TemporalLayer { layer: 0 }.path(),
            ReferenceTensor(vec![1, 3, 4]),
        ),
        (
            moshi::ObservationPoint::TextLogits.path(),
            ReferenceTensor(vec![1, 3, 13]),
        ),
        (
            moshi::ObservationPoint::DepthSliceLogits { slice: 0 }.path(),
            ReferenceTensor(vec![1, 1, 11]),
        ),
        (
            MODEL_LOGITS_OBSERVATION_PATH.to_owned(),
            ReferenceTensor(vec![1, 3, 19]),
        ),
    ]
    .into_iter()
    .collect();
    let summary = run_observation_scenario(replacements, None, 19);

    assert_eq!(summary.request_status, RequestStatus::Active);
    assert_eq!(summary.frontier, 1);
    assert!(summary.attention_offset > 0);
    assert_eq!(summary.random, 9);
    assert_eq!(summary.completion_calls, 1);
    assert_eq!(
        summary.paths,
        [
            moshi::ObservationPoint::TemporalInput.path(),
            moshi::ObservationPoint::TemporalLayer { layer: 0 }.path(),
            moshi::ObservationPoint::TextLogits.path(),
            moshi::ObservationPoint::DepthSliceLogits { slice: 0 }.path(),
            MODEL_LOGITS_OBSERVATION_PATH.to_owned(),
        ]
    );
    assert_eq!(summary.sampled_logits, [vec![1, 3, 13], vec![1, 1, 11]]);
    assert_eq!(summary.observed_values[1], ReferenceTensor(vec![1, 2, 4]));
    assert_eq!(summary.observed_values[2], ReferenceTensor(vec![1, 3, 17]));
}

#[allow(
    dead_code,
    reason = "owned by the unified reference_conformance target"
)]
pub(crate) fn realtime_observer_failure_discards_the_caller_owned_transaction() {
    let failed_path = MODEL_LOGITS_OBSERVATION_PATH.to_owned();
    let summary = run_observation_scenario(BTreeMap::new(), Some(failed_path.clone()), 17);

    assert_eq!(summary.request_status, RequestStatus::Failed);
    assert_eq!(summary.frontier, 0);
    assert_eq!(summary.attention_offset, 0);
    assert_eq!(summary.random, 7);
    assert_eq!(summary.completion_calls, 0);
    assert_eq!(summary.sampled_logits.len(), 2);
    assert_eq!(
        summary.paths,
        [
            moshi::ObservationPoint::TemporalInput.path(),
            moshi::ObservationPoint::TemporalLayer { layer: 0 }.path(),
            moshi::ObservationPoint::TextLogits.path(),
            moshi::ObservationPoint::DepthSliceLogits { slice: 0 }.path(),
            failed_path
        ]
    );
}

#[test]
fn selection_and_store_validation_precede_reference_construction() {
    let config = tiny_config();
    let store = Arc::new(MetadataCheckpointSource::from_config(&config));
    let prepared = selected(
        &config,
        &store,
        request(None, LayerWeightResidency::FullyResident),
    );
    let started = Rc::new(Cell::new(false));
    let error =
        moshi::visit_selected_moshi_realtime_architecture::<ReferenceBackend, ReferenceState, _>(
            prepared,
            Arc::new(MetadataCheckpointSource {
                tensors: BTreeMap::new(),
            }),
            &(),
            ConstructionVisitor {
                started: started.clone(),
                selection_complete: Rc::new(Cell::new(true)),
                ingress: moshi::realtime_ingress_contract(&config).unwrap(),
                residency: LayerWeightResidency::FullyResident,
                expects_transform: false,
                text_cardinality: config.text_vocabulary_size(),
                inspection_authority: None,
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("source catalog differs"));
    assert!(!started.get());
}

struct ReplicatedAgainstTensorParallelVisitor {
    config: MoshiConfig,
}

impl MoshiRealtimeArchitectureVisitor<ReferenceBackend, ReferenceState>
    for ReplicatedAgainstTensorParallelVisitor
{
    type Output = ();
    type Error = String;

    fn visit<A>(
        self,
        mut prepared: PreparedMoshiRealtimeArchitecture<A>,
        store: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: MoshiRealtimeExecutionArchitecture<ReferenceBackend, ReferenceState>
            + RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        assert!(prepared.take_parallel().is_some());
        let contract = prepared.take_contract();
        let replicated = moshi::LayeredModel::<ReferenceBackend>::new(self.config, &())
            .map_err(|error| error.to_string())?;
        let trace = Rc::new(RefCell::new(ConstructionTrace::default()));
        let error = construct_realtime_model::<_, ReferenceBackend, _>(
            replicated,
            None,
            contract,
            ReferenceConstructionMechanisms {
                trace,
                store,
                independent_authority: None,
            },
            &(),
        )
        .err()
        .expect("replicated architecture must not satisfy a selected TP contract");
        assert!(error.to_string().contains("differ from selection"));
        Ok(())
    }
}

#[test]
fn replicated_architecture_cannot_satisfy_tensor_parallel_selection() {
    let config =
        MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
    let store = Arc::new(MetadataCheckpointSource::from_config(&config));
    let prepared = selected(
        &config,
        &store,
        tensor_parallel_request(LayerWeightResidency::FullyResident),
    );
    moshi::visit_selected_moshi_realtime_architecture::<ReferenceBackend, ReferenceState, _>(
        prepared,
        store,
        &(),
        ReplicatedAgainstTensorParallelVisitor { config },
    )
    .unwrap();
}

#[allow(
    dead_code,
    reason = "owned by the unified reference_conformance target"
)]
pub(crate) fn native_moshi_resident_and_bounded_use_the_same_reference_frame_scheduler() {
    let resident = run_reference_scenario(tiny_config(), LayerWeightResidency::FullyResident, None);
    let bounded = run_reference_scenario(
        tiny_config(),
        LayerWeightResidency::LayerwiseHost(Default::default()),
        None,
    );
    assert_eq!(resident.unit_count, 2);
    assert_eq!(resident.state_layers, 2);
    assert_eq!(resident.task_count, bounded.task_count);
    assert_eq!(resident.unit_count, bounded.unit_count);
    assert_eq!(resident.state_layers, bounded.state_layers);
    assert_eq!(resident.random_after_sampled, 9);
    assert_eq!(resident.random_after_forced, 9);
    assert_eq!(bounded.random_after_sampled, 9);
    assert_eq!(bounded.random_after_forced, 9);
}

#[allow(
    dead_code,
    reason = "owned by the unified reference_conformance target"
)]
pub(crate) fn personaplex_uses_released_inspection_and_the_common_reference_scheduler() {
    let config =
        MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
    let summary = run_reference_scenario(config, LayerWeightResidency::FullyResident, None);
    assert_eq!(summary.state_layers, 38);
    assert_eq!(summary.unit_count, 48);
    assert!(summary.task_count > 600);
    assert_eq!(summary.random_after_sampled, 8);
    assert_eq!(summary.random_after_forced, 15);
}

#[test]
fn personaplex_prompt_plan_materializes_portable_frames_in_architecture_order() {
    use moshi::personaplex_prompt::{
        system_prompt_plan, AUDIO_TOKENS_PER_STREAM, SILENCE_TOKENS, SINE_TOKENS,
        TEXT_PADDING_TOKEN,
    };

    let voice = (0..AUDIO_TOKENS_PER_STREAM)
        .flat_map(|codebook| {
            [
                i32::try_from(100 + codebook * 10).unwrap(),
                i32::try_from(101 + codebook * 10).unwrap(),
            ]
        })
        .collect::<Vec<_>>();
    let text = vec![21, 22];
    let plan = system_prompt_plan(Some(&[1, AUDIO_TOKENS_PER_STREAM, 2]), &[1, 2]).unwrap();
    let frames =
        moshi::personaplex_prompt::materialize_prompt_frames(&plan, &voice, 2, &text, 2).unwrap();

    assert_eq!(frames.len(), 4);
    assert!(frames
        .iter()
        .all(|frame| frame.input_audio_tokens() == SINE_TOKENS));
    assert_eq!(
        frames[0].forced_generated_audio_tokens().unwrap(),
        &[100, 110, 120, 130, 140, 150, 160, 170]
    );
    assert_eq!(
        frames[1].forced_generated_audio_tokens().unwrap(),
        &[101, 111, 121, 131, 141, 151, 161, 171]
    );
    assert_eq!(
        frames[0].forced_text_tokens(),
        Some(&[TEXT_PADDING_TOKEN][..])
    );
    assert_eq!(
        frames[1].forced_text_tokens(),
        Some(&[TEXT_PADDING_TOKEN][..])
    );
    assert_eq!(
        frames[2].forced_generated_audio_tokens(),
        Some(&SILENCE_TOKENS[..])
    );
    assert_eq!(
        frames[3].forced_generated_audio_tokens(),
        Some(&SILENCE_TOKENS[..])
    );
    assert_eq!(frames[2].forced_text_tokens(), Some(&[21][..]));
    assert_eq!(frames[3].forced_text_tokens(), Some(&[22][..]));
}

#[allow(
    dead_code,
    reason = "owned by the unified reference_conformance target"
)]
pub(crate) fn load_time_affine_transform_reaches_typed_construction_and_frame_execution() {
    let summary = run_reference_scenario(
        quantizable_native_config(),
        LayerWeightResidency::FullyResident,
        Some(QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        }),
    );
    assert_eq!(summary.unit_count, 2);
}

#[derive(Default)]
struct ArtifactSourceCounters {
    metadata_reads: AtomicUsize,
    provenance_reads: AtomicUsize,
    payload_reads: AtomicUsize,
}

impl ArtifactSourceCounters {
    fn reset(&self) {
        self.metadata_reads.store(0, Ordering::SeqCst);
        self.provenance_reads.store(0, Ordering::SeqCst);
        self.payload_reads.store(0, Ordering::SeqCst);
    }
}

struct IndependentArtifactSource {
    inner: MemoryWeightStore,
    metadata: Mutex<TensorMetadata>,
    counters: Arc<ArtifactSourceCounters>,
}

impl IndependentArtifactSource {
    fn new(counters: Arc<ArtifactSourceCounters>) -> Self {
        let bytes = [1.0_f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let inner = MemoryWeightStore::from_safetensors([(
            "weight".into(),
            safetensors::Dtype::F32,
            vec![2],
            bytes,
        )])
        .unwrap();
        Self {
            inner,
            metadata: Mutex::new(TensorMetadata {
                name: "weight".into(),
                logical_shape: vec![2],
                physical_shape: vec![2],
                stored_dtype: StoredDtype::F32,
                encoded_byte_len: 8,
                backing_shard: None,
            }),
            counters,
        }
    }

    fn replace_metadata(&self, metadata: TensorMetadata) {
        *self.metadata.lock().unwrap() = metadata;
    }
}

impl CheckpointSource for IndependentArtifactSource {
    fn source_keys(&self) -> Vec<String> {
        vec!["weight".into()]
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.counters.metadata_reads.fetch_add(1, Ordering::SeqCst);
        if key != "weight" {
            return Err(StoreError::UnknownTensor { key: key.into() });
        }
        Ok(self.metadata.lock().unwrap().clone())
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.counters.payload_reads.fetch_add(1, Ordering::SeqCst);
        self.inner.acquire_lease(request)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.inner.source_diagnostics()
    }

    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, StoreError> {
        self.counters
            .provenance_reads
            .fetch_add(1, Ordering::SeqCst);
        let metadata = self.source_metadata(key)?;
        Ok(eredu_checkpoint::store::TensorSourceProvenance {
            catalog_key: key.into(),
            physical_tensor: key.into(),
            output: key.into(),
            backing_shard: metadata.backing_shard,
            source_encoding: eredu_checkpoint::SourceTensorEncoding::Safetensors(
                metadata.stored_dtype,
            ),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct IndependentInspectionResolver;

impl eredu_core::ModelConfigurationResolver for IndependentInspectionResolver {
    type ArtifactPlan = ();

    fn resolve_safetensors(
        &self,
        json: &serde_json::Value,
    ) -> Result<eredu_core::ResolvedModelConfiguration<()>, eredu_core::artifact::ArtifactError>
    {
        let model_type = json
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                eredu_core::artifact::ArtifactError::InvalidArtifact(
                    "independent fixture requires model_type".into(),
                )
            })?;
        Ok(eredu_core::ResolvedModelConfiguration::new(
            eredu_core::ModelConfiguration::new(
                model_type,
                model_type,
                "independent-reference",
                eredu_core::LoadingProtocol::Model,
                Some(json.clone()),
            )?,
            (),
        ))
    }

    fn resolve_gguf(
        &self,
        _architecture: &str,
        _checkpoint: &eredu_gguf::Checkpoint,
    ) -> Result<eredu_core::ResolvedModelConfiguration<()>, eredu_core::artifact::ArtifactError>
    {
        Err(eredu_core::artifact::ArtifactError::InvalidArtifact(
            "independent fixture admits safetensors only".into(),
        ))
    }

    fn gguf_companion_requirements(
        &self,
        _architecture: &str,
        _checkpoint: &eredu_gguf::Checkpoint,
    ) -> Result<Vec<eredu_core::GgufCompanionRequirement>, eredu_core::artifact::ArtifactError>
    {
        Ok(Vec::new())
    }
}

#[derive(Default)]
struct BoundedAdapterCounters {
    artifact_identity: AtomicUsize,
    image_processing: AtomicUsize,
    audio_processing: AtomicUsize,
    inspection_admission: AtomicUsize,
    placement_communication: AtomicUsize,
    prepared_source: AtomicUsize,
    binding: AtomicUsize,
    session: AtomicUsize,
}

#[derive(Default)]
struct BoundedIndependentAdapter {
    counters: BoundedAdapterCounters,
}

impl BoundedIndependentAdapter {
    fn run(&self) {
        let first = tempfile::tempdir().unwrap();
        let relocated = tempfile::tempdir().unwrap();
        let first_path = first.path().join("model.safetensors");
        let relocated_path = relocated.path().join("model.safetensors");
        std::fs::write(&first_path, b"exact checkpoint bytes").unwrap();
        std::fs::write(&relocated_path, b"exact checkpoint bytes").unwrap();
        let identity = fingerprint_filesystem_artifact(
            "independent-reference",
            [ArtifactFile::new("weights", &first_path)],
        )
        .unwrap();
        assert_eq!(
            identity,
            fingerprint_filesystem_artifact(
                "independent-reference",
                [ArtifactFile::new("weights", &relocated_path)],
            )
            .unwrap()
        );
        std::fs::write(&relocated_path, b"changed checkpoint content").unwrap();
        assert_ne!(
            identity,
            fingerprint_filesystem_artifact(
                "independent-reference",
                [ArtifactFile::new("weights", &relocated_path)],
            )
            .unwrap()
        );
        self.counters
            .artifact_identity
            .fetch_add(1, Ordering::SeqCst);

        let image = eredu_media::image::RgbImageView::packed(&[10, 20, 30], 1, 1).unwrap();
        let resized = eredu_media::image::resize_rgb8_bicubic(image, 2, 2).unwrap();
        assert_eq!(
            resized.pixels(),
            &[10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
        );
        self.counters
            .image_processing
            .fetch_add(1, Ordering::SeqCst);
        let waveform = eredu_media::audio::AudioWaveform::new(
            &[0.0, 0.25, -0.25, 0.5, 0.0, -0.5, 0.25, 0.0],
            8,
        )
        .unwrap();
        let audio = eredu_media::audio::extract_log_mel(
            waveform,
            &eredu_media::audio::LogMelConfig {
                sample_rate: 8,
                frame_length: 4,
                hop_length: 2,
                fft_length: 4,
                mel_bins: 2,
                min_frequency: 0.0,
                max_frequency: 4.0,
                mel_floor: 1e-6,
                max_samples: 8,
                pad_to_multiple: 2,
            },
        )
        .unwrap();
        assert_eq!(audio.values.len(), audio.frames * audio.mel_bins);
        assert!(audio.values.iter().all(|value| value.is_finite()));
        self.counters
            .audio_processing
            .fetch_add(1, Ordering::SeqCst);

        std::fs::write(
            first.path().join("config.json"),
            br#"{"model_type":"independent_reference"}"#,
        )
        .unwrap();
        let tensor_bytes = [1.0_f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let tensor =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2], &tensor_bytes)
                .unwrap();
        safetensors::tensor::serialize_to_file(
            [("weight", tensor)],
            None,
            &first.path().join("model.safetensors"),
        )
        .unwrap();
        let artifact_inspection =
            eredu_core::inspect_artifact(first.path(), &IndependentInspectionResolver).unwrap();

        let policy =
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::LayerwiseHost)
                .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
        let request = eredu_core::PreparationAdmissionRequest::new(
            eredu_core::LoadingProtocol::Model,
            eredu_core::ArtifactFormat::SafeTensors,
            policy,
            eredu_core::ArchitecturePreparationCapabilities::new(
                false,
                false,
                false,
                false,
                false,
                eredu_core::InputModalities {
                    text: true,
                    image: false,
                    audio: true,
                    video: false,
                },
            ),
        )
        .with_exact_completion(true);
        let mechanisms = eredu_core::PreparationMechanismCapabilities::new(true, false)
            .with_residency(eredu_core::ResidencyRequest::LayerwiseHost, true)
            .with_input_modalities(eredu_core::InputModalities {
                text: true,
                image: false,
                audio: true,
                video: false,
            })
            .with_exact_completion(true)
            .with_session(SessionCapabilities::new(true, true, true));
        let admission = eredu_core::admit_preparation(request, mechanisms).unwrap();
        let admitted_report = eredu_core::assemble_portable_model_inspection(
            &artifact_inspection,
            admission,
            eredu_core::InputModalities {
                text: true,
                image: false,
                audio: true,
                video: false,
            },
            None,
            Some((
                true,
                eredu_core::MediaFeatureAvailability {
                    image: true,
                    audio: true,
                },
            )),
        );
        assert_eq!(admitted_report.preparation_admission, Some(admission));
        self.counters
            .inspection_admission
            .fetch_add(1, Ordering::SeqCst);

        let rank = eredu_runtime::PlacementRank::new(4, 0).unwrap();
        let mut placement = eredu_runtime::PlacementPlan::new(rank);
        placement
            .insert_expected(
                "weight",
                vec![2],
                eredu_runtime::TensorPlacement::Shard {
                    axis: 0,
                    index: 0,
                    parts: 2,
                },
            )
            .unwrap();
        let selection = match placement.resolve("weight", &[2]).unwrap() {
            eredu_runtime::ResolvedTensorPlacement::Selection(selection) => selection,
            other => panic!("expected a nontrivial placed selection, got {other:?}"),
        };
        let topology = ParallelTopology::new(2, 2, 1, 1).unwrap();
        let all_reduce = eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::AllReduceSum,
            [eredu_core::checkpoint::TensorDtype::F32],
            eredu_runtime::CommunicationTensorLimits::new(1, 2, 1024, None).unwrap(),
            true,
        )
        .unwrap();
        let send_receive = eredu_runtime::CommunicationOperationRequirement::tensors(
            eredu_runtime::CommunicationOperation::SendReceive,
            [eredu_core::checkpoint::TensorDtype::F32],
            eredu_runtime::CommunicationTensorLimits::new(1, 2, 1024, None).unwrap(),
            true,
        )
        .unwrap();
        let communication = eredu_runtime::TopologyCommunicationPlan::new()
            .with_completion_policy(
                CommunicationCompletionPolicy::new(
                    Duration::from_secs(1),
                    CompletionCancellationMode::QuarantineUntilComplete,
                )
                .unwrap(),
            )
            .with_tensor_groups(
                eredu_runtime::CommunicationGroupRequirements::new([all_reduce.clone()]).unwrap(),
            )
            .with_pipeline_routes(send_receive.clone())
            .unwrap();
        let manifests =
            eredu_runtime::project_all_communication_manifests(topology, &communication).unwrap();
        eredu_runtime::validate_compatible_communication_manifests(&manifests).unwrap();
        assert_eq!(manifests.len(), 4);
        assert!(manifests
            .iter()
            .all(|manifest| !manifest.groups().is_empty()));
        assert!(manifests
            .iter()
            .all(|manifest| !manifest.routes().is_empty()));
        let communication_capabilities =
            eredu_runtime::CommunicationCapabilities::new([all_reduce, send_receive])
                .unwrap()
                .with_completion_capabilities(
                    CommunicationCompletionCapabilities::new([
                        CompletionCancellationMode::QuarantineUntilComplete,
                    ])
                    .unwrap(),
                );
        let prepared_communication = eredu_runtime::prepare_communication_realization(
            &manifests[0],
            &manifests,
            &communication_capabilities,
            eredu_runtime::CommunicationTopologyCapabilities::FullyConnected,
        )
        .unwrap();
        self.counters
            .placement_communication
            .fetch_add(1, Ordering::SeqCst);

        let counters = Arc::new(ArtifactSourceCounters::default());
        let source = Arc::new(IndependentArtifactSource::new(Arc::clone(&counters)));
        let metadata = source.source_metadata("weight").unwrap();
        let provenance = source.source_provenance("weight").unwrap();
        counters.reset();
        clear_reference_trace();
        let shared: SharedCheckpointSource = source.clone();
        let prepared = PreparedCheckpointSource::new(
            shared,
            BTreeMap::from([(
                "weight".into(),
                PreparedTensorSource {
                    metadata: metadata.clone(),
                    provenance,
                },
            )]),
        )
        .unwrap();
        self.counters.prepared_source.fetch_add(1, Ordering::SeqCst);
        assert!(counters.metadata_reads.load(Ordering::SeqCst) > 0);
        assert!(counters.provenance_reads.load(Ordering::SeqCst) > 0);
        assert_eq!(counters.payload_reads.load(Ordering::SeqCst), 0);
        assert_eq!(reference_trace().parameter_materializations, 0);

        let mut substituted = metadata.clone();
        substituted.logical_shape = vec![1, 2];
        source.replace_metadata(substituted);
        counters.reset();
        clear_reference_trace();
        let binding = WeightBinding::new("weight", "weight", selection.clone(), 4).unwrap();
        assert!(materialize_bindings::<ReferenceBackend>(
            &prepared,
            std::slice::from_ref(&binding),
            &()
        )
        .is_err());
        assert!(counters.metadata_reads.load(Ordering::SeqCst) > 0);
        assert_eq!(counters.payload_reads.load(Ordering::SeqCst), 0);
        assert_eq!(reference_trace().parameter_materializations, 0);

        source.replace_metadata(metadata);
        counters.reset();
        clear_reference_trace();
        let materialized = materialize_bindings::<ReferenceBackend>(&prepared, &[binding], &())
            .expect("independent backend materializes through the canonical prepared source");
        assert_eq!(materialized.len(), 1);
        assert_eq!(counters.payload_reads.load(Ordering::SeqCst), 1);
        assert_eq!(reference_trace().parameter_materializations, 1);
        self.counters.binding.fetch_add(1, Ordering::SeqCst);
        let mut bound_module = IndependentBoundModule {
            weight: ReferenceTensor(vec![1]),
        };
        bind_materialized_unit::<ReferenceBackend, _>(&mut bound_module, materialized).unwrap();
        assert_eq!(bound_module.weight, ReferenceTensor(vec![1]));
        let make_resources = |causal_value, omit_route_resources| {
            Rc::new(IndependentAuthorityResources {
                bound_module: bound_module.clone(),
                placement: selection.clone(),
                communication: prepared_communication.clone(),
                construction_uses: Cell::new(0),
                group_callbacks: Cell::new(0),
                route_callbacks: Cell::new(0),
                bound_value_trace: RefCell::new(Vec::new()),
                causal_value,
                executed_outputs: RefCell::new(Vec::new()),
                group_resources: RefCell::new(Vec::new()),
                route_resources: RefCell::new(Vec::new()),
                omit_route_resources,
            })
        };
        let resources = make_resources(2, false);

        let realized = Rc::new(RefCell::new(None));
        let summary = run_reference_scenario_with_authority(
            tiny_config(),
            LayerWeightResidency::LayerwiseHost(Default::default()),
            None,
            Some(IndependentSessionAuthority {
                inspection: artifact_inspection.clone(),
                admission,
                admitted_report: admitted_report.clone(),
                resources: resources.clone(),
                realized: realized.clone(),
            }),
        );
        assert_eq!(summary.state_layers, 2);
        let realized = realized
            .borrow_mut()
            .take()
            .expect("the constructed and executed session must publish realized authority");
        assert_eq!(realized.capabilities, admission.session_capabilities());
        assert_eq!(realized.report, admitted_report);
        assert!(resources.construction_uses.get() > 0);
        assert_eq!(
            resources.group_callbacks.get(),
            resources.communication.manifest().groups().len()
        );
        assert_eq!(
            resources.route_callbacks.get(),
            resources.communication.manifest().routes().len()
        );
        assert_eq!(resources.bound_value_trace.borrow().as_slice(), &[1, 1]);
        assert_eq!(
            realized.scheduler_capabilities.limits.max_active_requests,
            1
        );
        assert!(!realized
            .scheduler_capabilities
            .non_preemptible_interval
            .is_empty());
        assert!(realized.scheduler_report.submitted_work > 0);
        assert_eq!(realized.executed_outputs.len(), 2);

        let alternative_resources = make_resources(3, false);
        let alternative_realized = Rc::new(RefCell::new(None));
        let alternative_summary = run_reference_scenario_with_authority(
            tiny_config(),
            LayerWeightResidency::LayerwiseHost(Default::default()),
            None,
            Some(IndependentSessionAuthority {
                inspection: artifact_inspection.clone(),
                admission,
                admitted_report: admitted_report.clone(),
                resources: alternative_resources.clone(),
                realized: alternative_realized.clone(),
            }),
        );
        let alternative_realized = alternative_realized.borrow_mut().take().unwrap();
        assert_ne!(
            summary.random_after_sampled,
            alternative_summary.random_after_sampled
        );
        assert_ne!(
            summary.random_after_forced,
            alternative_summary.random_after_forced
        );
        assert_eq!(alternative_realized.executed_outputs.len(), 2);
        assert_eq!(alternative_realized.report, admitted_report);
        assert!(alternative_resources.construction_uses.get() > 0);
        assert_eq!(
            alternative_resources.group_callbacks.get(),
            alternative_resources
                .communication
                .manifest()
                .groups()
                .len()
        );
        assert_eq!(
            alternative_resources.route_callbacks.get(),
            alternative_resources
                .communication
                .manifest()
                .routes()
                .len()
        );
        let missing_resources = make_resources(4, true);
        let missing_realized = Rc::new(RefCell::new(None));
        let error = try_run_reference_scenario_with_authority(
            tiny_config(),
            LayerWeightResidency::LayerwiseHost(Default::default()),
            None,
            Some(IndependentSessionAuthority {
                inspection: artifact_inspection,
                admission,
                admitted_report: admitted_report.clone(),
                resources: missing_resources.clone(),
                realized: missing_realized.clone(),
            }),
        )
        .unwrap_err();
        assert!(error.contains("prepared communication resources are incomplete"));
        assert!(missing_realized.borrow().is_none());
        assert!(missing_resources.route_resources.borrow().is_empty());
        self.counters.session.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn bounded_independent_adapter_consumes_the_complete_neutral_preparation_stack() {
    let adapter = BoundedIndependentAdapter::default();
    adapter.run();
    assert_eq!(adapter.counters.artifact_identity.load(Ordering::SeqCst), 1);
    assert_eq!(adapter.counters.image_processing.load(Ordering::SeqCst), 1);
    assert_eq!(adapter.counters.audio_processing.load(Ordering::SeqCst), 1);
    assert_eq!(
        adapter.counters.inspection_admission.load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        adapter
            .counters
            .placement_communication
            .load(Ordering::SeqCst),
        1
    );
    assert_eq!(adapter.counters.prepared_source.load(Ordering::SeqCst), 1);
    assert_eq!(adapter.counters.binding.load(Ordering::SeqCst), 1);
    assert_eq!(adapter.counters.session.load(Ordering::SeqCst), 1);
}

#[cfg(test)]
mod unified_conformance_compatibility_wrappers {
    use super::*;

    fn run_on_reference_stack(case: fn()) {
        std::thread::Builder::new()
            .name("reference-realtime-compatibility-wrapper".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(case)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn observer_failure_transaction() {
        run_on_reference_stack(realtime_observer_failure_discards_the_caller_owned_transaction);
    }

    #[test]
    fn moshi_resident_and_bounded() {
        run_on_reference_stack(
            native_moshi_resident_and_bounded_use_the_same_reference_frame_scheduler,
        );
    }

    #[test]
    fn personaplex_reference_scheduler() {
        run_on_reference_stack(
            personaplex_uses_released_inspection_and_the_common_reference_scheduler,
        );
    }

    #[test]
    fn affine_transform_construction() {
        run_on_reference_stack(
            load_time_affine_transform_reaches_typed_construction_and_frame_execution,
        );
    }
}
