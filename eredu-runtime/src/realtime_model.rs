//! Family-blind construction of a selected layered realtime model.

use eredu_core::checkpoint_mechanisms::CheckpointParameterBackend;
use eredu_core::SubmissionBackend;

use std::{collections::BTreeMap, marker::PhantomData};

use eredu_checkpoint::quantization::BoundedQuantizationTarget;
use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeMetadata};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime, ParameterGroupOwner,
    RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, RuntimeState, SelectedRealtimeRealization,
};
use eredu_core::residency::parameter_binding::{WeightBinding, WeightBindingPlan};

/// Architecture-issued identities that bind concrete modules to one selection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeArchitectureConstructionIdentity {
    architecture: crate::RealtimeIdentity,
    speech_schedule: crate::RealtimeIdentity,
    state_layout: crate::RealtimeIdentity,
}

impl RealtimeArchitectureConstructionIdentity {
    /// Creates one exact construction witness.
    pub fn new(
        architecture: crate::RealtimeIdentity,
        speech_schedule: crate::RealtimeIdentity,
        state_layout: crate::RealtimeIdentity,
    ) -> Self {
        Self {
            architecture,
            speech_schedule,
            state_layout,
        }
    }
}

/// Supplies stable semantic identities for a concrete neutral architecture.
pub trait RealtimeArchitectureIdentity {
    /// Returns the identities derived from the architecture's normalized policy.
    fn realtime_construction_identity(
        &self,
    ) -> Result<RealtimeArchitectureConstructionIdentity, String>;
}

impl<D, K, R> RealtimeArchitectureIdentity for crate::ComponentAdapter<D, K, R>
where
    D: std::borrow::Borrow<crate::ComponentDefinition>,
    K: RealtimeArchitectureIdentity,
{
    fn realtime_construction_identity(
        &self,
    ) -> Result<RealtimeArchitectureConstructionIdentity, String> {
        self.kernel().realtime_construction_identity()
    }
}

/// One selected component paired with its exact architecture recipe payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeMaterializationComponent {
    requirement: RealtimeWeightComponentRequirement,
    source_provenance: Vec<eredu_checkpoint::store::TensorSourceProvenance>,
}

impl RealtimeMaterializationComponent {
    /// Pairs an exact selected component with its admission-time physical sources.
    pub fn new(
        requirement: RealtimeWeightComponentRequirement,
        source_provenance: impl IntoIterator<Item = eredu_checkpoint::store::TensorSourceProvenance>,
    ) -> Result<Self, RealtimeModelContractError> {
        let source_provenance = source_provenance.into_iter().collect::<Vec<_>>();
        if requirement.source_occurrences().len() != source_provenance.len()
            || requirement
                .source_occurrences()
                .iter()
                .zip(&source_provenance)
                .any(|(expected, actual)| expected.as_str() != actual.catalog_key)
        {
            return Err(RealtimeModelContractError::SourceProvenanceMismatch {
                target: requirement.target().as_str().to_owned(),
            });
        }
        Ok(Self {
            requirement,
            source_provenance,
        })
    }

    /// Returns the exact selected component requirement.
    pub const fn requirement(&self) -> &RealtimeWeightComponentRequirement {
        &self.requirement
    }

    /// Returns the architecture recipe for a source-backed component.
    pub const fn recipe(&self) -> Option<&DerivedWeightRecipe> {
        self.requirement.recipe()
    }

    /// Returns admission-time recipe output metadata.
    pub const fn recipe_output(&self) -> Option<&RecipeMetadata> {
        self.requirement.recipe_output()
    }

    /// Returns exact physical source provenance in recipe traversal order.
    pub fn source_provenance(&self) -> &[eredu_checkpoint::store::TensorSourceProvenance] {
        &self.source_provenance
    }
}

/// Exact materialization work for one selected logical parameter target.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeMaterializationTask {
    lowering: RealtimeWeightLoweringRequirement,
    owner: ParameterGroupOwner,
    components: Vec<RealtimeMaterializationComponent>,
}

impl RealtimeMaterializationTask {
    /// Creates a task whose recipe payload exactly covers the selected components.
    pub fn new(
        lowering: RealtimeWeightLoweringRequirement,
        owner: ParameterGroupOwner,
        components: impl IntoIterator<Item = RealtimeMaterializationComponent>,
    ) -> Result<Self, RealtimeModelContractError> {
        let components = components.into_iter().collect::<Vec<_>>();
        let expected = lowering.components();
        if components.len() != expected.len()
            || components
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.requirement() != expected)
        {
            return Err(RealtimeModelContractError::ComponentCoverageMismatch {
                target: lowering.target().as_str().to_owned(),
            });
        }
        let primary = components
            .iter()
            .find(|component| {
                component.requirement().role() == RealtimeWeightComponentRole::Primary
            })
            .expect("selected lowering validation guarantees one primary component");
        if primary
            .recipe_output()
            .is_some_and(|output| output.shape != lowering.descriptor().physical_shape())
        {
            return Err(RealtimeModelContractError::RecipeGeometryMismatch {
                target: lowering.target().as_str().to_owned(),
            });
        }
        if let (Some(output), eredu_checkpoint::SourceTensorEncoding::Safetensors(stored)) =
            (primary.recipe_output(), lowering.descriptor().source())
        {
            if output.dtype != eredu_checkpoint::recipe::RecipeDtype::from(stored.clone()) {
                return Err(RealtimeModelContractError::RecipeEncodingMismatch {
                    target: lowering.target().as_str().to_owned(),
                });
            }
        }
        Ok(Self {
            lowering,
            owner,
            components,
        })
    }

    /// Returns the selected lowering and target identity.
    pub const fn lowering(&self) -> &RealtimeWeightLoweringRequirement {
        &self.lowering
    }

    /// Returns the exact static or execution-unit owner.
    pub const fn owner(&self) -> &ParameterGroupOwner {
        &self.owner
    }

    /// Returns ordered primary and companion recipe payloads.
    pub fn components(&self) -> &[RealtimeMaterializationComponent] {
        &self.components
    }
}

/// Canonical owner-partitioned bindings derived from exact realtime tasks.
struct RealtimeTaskBindingPlan {
    pinned: Vec<WeightBinding>,
    units: BTreeMap<ParameterGroupOwner, Vec<WeightBinding>>,
}

/// Projects the selected transforms into exact checkpoint recipes and output names.
fn realtime_quantization_targets(
    tasks: &[RealtimeMaterializationTask],
    quantization: eredu_checkpoint::WeightQuantization,
) -> Result<Vec<BoundedQuantizationTarget>, String> {
    let mut targets = BTreeMap::new();
    for lowering in tasks
        .iter()
        .map(RealtimeMaterializationTask::lowering)
        .filter(|lowering| lowering.kind().transforms_payload())
    {
        let target = lowering.target().as_str();
        if lowering.descriptor().executable().weight_quantization() != Some(quantization) {
            return Err(format!(
                "selected realtime transform {target:?} does not match its executable format"
            ));
        }
        let scale = lowering.scale().ok_or_else(|| {
            format!("selected realtime transform {target:?} has no scale component")
        })?;
        let recipe = lowering.primary().recipe().ok_or_else(|| {
            format!("selected realtime transform {target:?} has no source recipe")
        })?;
        let binding = BoundedQuantizationTarget::from_recipe(
            target,
            scale.target().as_str(),
            lowering
                .affine_bias()
                .map(|component| component.target().as_str()),
            recipe.clone(),
        )
        .map_err(|error| error.to_string())?;
        if targets.insert(target, binding).is_some() {
            return Err(format!(
                "selected realtime transform {target:?} appears more than once"
            ));
        }
    }
    if targets.is_empty() {
        return Err("selected realtime transform contains no packed matrix tasks".into());
    }
    Ok(targets.into_values().collect())
}

/// Validates every exact realtime task and backend recipe capability without payload access.
///
/// This gate is valid against the original checkpoint even when later transform
/// work publishes authoritative packed outputs.
fn preflight_realtime_materialization_tasks<B: CheckpointParameterBackend>(
    tasks: &[RealtimeMaterializationTask],
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<(), RealtimeModelContractError> {
    for task in tasks {
        let transformed = task.lowering().kind().transforms_payload();
        let targets = task
            .components()
            .iter()
            .map(|component| component.requirement().target().as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for component in task.components() {
            for admitted in component.source_provenance() {
                let actual = source
                    .source_provenance(&admitted.catalog_key)
                    .map_err(|error| RealtimeModelContractError::BindingPlan {
                        detail: error.to_string(),
                    })?;
                if &actual != admitted {
                    return Err(RealtimeModelContractError::BindingPlan {
                        detail: format!(
                            "realtime component {:?} differs from admitted source provenance",
                            component.requirement().target().as_str()
                        ),
                    });
                }
            }
            if let Some(owner) = component.requirement().recipe_owner() {
                if owner != component.requirement().target() && !targets.contains(owner.as_str()) {
                    return Err(RealtimeModelContractError::BindingPlan {
                        detail: format!("realtime alias owner {:?} is absent", owner.as_str()),
                    });
                }
            }
            if let Some(recipe) = component.recipe() {
                let actual = recipe.infer(source).map_err(|error| {
                    RealtimeModelContractError::BindingPlan {
                        detail: error.to_string(),
                    }
                })?;
                if component.recipe_output() != Some(&actual) {
                    return Err(RealtimeModelContractError::BindingPlan {
                        detail: format!(
                            "realtime component {:?} recipe output drifted",
                            component.requirement().target().as_str()
                        ),
                    });
                }
                B::preflight_recipe(recipe, source).map_err(|error| {
                    RealtimeModelContractError::BindingPlan {
                        detail: error.to_string(),
                    }
                })?;
            } else if component.requirement().recipe_owner().is_none() && !transformed {
                return Err(RealtimeModelContractError::BindingPlan {
                    detail: format!(
                        "realtime component {:?} has neither recipe nor alias owner",
                        component.requirement().target().as_str()
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Derives the singular canonical binding partitions from exact realtime tasks.
fn realtime_task_binding_plan(
    tasks: &[RealtimeMaterializationTask],
    source: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<RealtimeTaskBindingPlan, RealtimeModelContractError> {
    let mut pinned = Vec::new();
    let mut units = BTreeMap::<ParameterGroupOwner, Vec<WeightBinding>>::new();
    for task in tasks {
        let destination = match task.owner() {
            ParameterGroupOwner::StaticRole(_)
            | ParameterGroupOwner::StaticAnyOf(_)
            | ParameterGroupOwner::StaticUnitConsumers { .. } => &mut pinned,
            ParameterGroupOwner::ExecutionUnit { .. } => {
                units.entry(task.owner().clone()).or_default()
            }
        };
        let transformed = task.lowering().kind().transforms_payload();
        for component in task.components() {
            let requirement = component.requirement();
            let target = requirement.target().as_str();
            let binding = if transformed {
                if !source.is_authoritative_materialized_key(target) {
                    return Err(RealtimeModelContractError::BindingPlan {
                        detail: format!(
                            "transformed realtime output {target:?} is not authoritative"
                        ),
                    });
                }
                let metadata = source.source_metadata(target).map_err(|error| {
                    RealtimeModelContractError::BindingPlan {
                        detail: error.to_string(),
                    }
                })?;
                WeightBinding::new(
                    target,
                    target,
                    eredu_checkpoint::store::TensorSelection::Full,
                    metadata.encoded_byte_len,
                )
            } else {
                let output = component.recipe_output().ok_or_else(|| {
                    RealtimeModelContractError::BindingPlan {
                        detail: format!("realtime recipe component {target:?} has no output"),
                    }
                })?;
                match requirement.recipe_owner() {
                    Some(owner) if owner != requirement.target() => {
                        WeightBinding::alias(target, owner.as_str(), output.byte_len())
                    }
                    _ => WeightBinding::from_recipe(
                        target,
                        component.recipe().cloned().ok_or_else(|| {
                            RealtimeModelContractError::BindingPlan {
                                detail: format!("realtime component {target:?} has no recipe"),
                            }
                        })?,
                        output.byte_len(),
                    ),
                }
            }
            .map_err(|error| RealtimeModelContractError::BindingPlan {
                detail: error.to_string(),
            })?
            .with_logical_target(task.lowering().target().as_str())
            .map_err(|error| RealtimeModelContractError::BindingPlan {
                detail: error.to_string(),
            })?;
            destination.push(binding);
        }
    }
    WeightBindingPlan::new(&pinned).map_err(|error| RealtimeModelContractError::BindingPlan {
        detail: error.to_string(),
    })?;
    for bindings in units.values() {
        WeightBindingPlan::new(bindings).map_err(|error| {
            RealtimeModelContractError::BindingPlan {
                detail: error.to_string(),
            }
        })?;
    }
    Ok(RealtimeTaskBindingPlan { pinned, units })
}

/// Selected realization paired with complete architecture recipe payloads.
#[derive(Debug)]
pub struct PreparedRealtimeModelContract {
    selected: SelectedRealtimeRealization,
    tasks: Vec<RealtimeMaterializationTask>,
}

impl PreparedRealtimeModelContract {
    /// Validates that one exact task exists for every selected target.
    pub fn new(
        selected: SelectedRealtimeRealization,
        tasks: impl IntoIterator<Item = RealtimeMaterializationTask>,
    ) -> Result<Self, RealtimeModelContractError> {
        let tasks = tasks.into_iter().collect::<Vec<_>>();
        if tasks.len() != selected.weight_lowerings().len() {
            return Err(RealtimeModelContractError::TaskCoverageMismatch);
        }
        for (task, selected_lowering) in tasks.iter().zip(selected.weight_lowerings()) {
            if selected_lowering != &task.lowering {
                return Err(RealtimeModelContractError::TaskSelectionMismatch {
                    target: task.lowering.target().as_str().to_owned(),
                });
            }
            let expected_owner = selected
                .execution_parameters()
                .groups()
                .iter()
                .find_map(|group| {
                    group
                        .group()
                        .members()
                        .iter()
                        .any(|member| member.target() == task.lowering.target().as_str())
                        .then(|| group.owner())
                })
                .ok_or_else(|| RealtimeModelContractError::TaskOwnerUnavailable {
                    target: task.lowering.target().as_str().to_owned(),
                })?;
            if expected_owner != &task.owner {
                return Err(RealtimeModelContractError::TaskOwnerMismatch {
                    target: task.lowering.target().as_str().to_owned(),
                });
            }
        }
        Ok(Self { selected, tasks })
    }

    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    /// Returns exact materialization work in architecture order.
    pub fn tasks(&self) -> &[RealtimeMaterializationTask] {
        &self.tasks
    }

    /// Consumes the contract into selection and exact work.
    pub fn into_parts(
        self,
    ) -> (
        SelectedRealtimeRealization,
        Vec<RealtimeMaterializationTask>,
    ) {
        (self.selected, self.tasks)
    }
}

/// Generic mechanisms used only to materialize and store a selected model.
pub trait RealtimeModelConstructionMechanisms<A, B>:
    eredu_core::checkpoint_mechanisms::unit_binding::ParameterStorageBinding<B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Concrete architecture state realization.
    type State: RuntimeState<B>;
    /// Shared resident/bounded policy failure.
    type PolicyError;
    /// Fully resident unit storage.
    type ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>
        + crate::bounded_execution::units::BindResidentPolicy<
            B,
            A::Unit,
            crate::ParameterUnitStorage<B, Self>,
        >;
    /// Host-windowed or disk-streamed unit storage.
    type BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>
        + crate::bounded_execution::units::BindBoundedPolicy<
            A::Unit,
            crate::ParameterUnitStorage<B, Self>,
        >;
    /// Mechanism failure.
    type Error: From<eredu_core::BackendFailure>;

    /// Native storage geometry and encoding for exact portable targets.
    type ParameterEncoding: eredu_core::checkpoint_mechanisms::ExactParameterMaterializer<B>;
}

impl<A, B, M> RealtimeModelConstructionMechanisms<A, B> for M
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    B: eredu_core::backend::LocalComponentStorage<
        CacheStorage: eredu_core::cache::mechanisms::ResidentKeyValueStorage
                          + eredu_core::cache::mechanisms::PagedKeyValueStorage,
    >,
    A: LayeredArchitecture<B, crate::state::KeyValueState<B::CacheStorage>, Unit: 'static>,
    M: crate::ParameterUnitMechanisms<B>,
    M::Parameters: eredu_core::checkpoint_mechanisms::ExactParameterMaterializer<B>,
    M::UnitPolicy<A::Unit>: LayerwisePolicy<B, A::Unit, Error: From<eredu_core::BackendFailure>>,
{
    type State = crate::state::KeyValueState<B::CacheStorage>;
    type PolicyError = <M::UnitPolicy<A::Unit> as LayerwisePolicy<B, A::Unit>>::Error;
    type ResidentPolicy = M::UnitPolicy<A::Unit>;
    type BoundedPolicy = M::UnitPolicy<A::Unit>;
    type Error = Self::PolicyError;
    type ParameterEncoding = M::Parameters;
}

/// Retained portable construction that can be bound by native mechanisms.
pub trait PreparedRealtimeBinding<B, State, M>: Sized {
    /// Bound portable execution.
    type Execution;
    /// Failure constructing or binding this preparation.
    type Error;

    /// Exact admitted realization used to validate the native target.
    fn realization(&self) -> &SelectedRealtimeRealization;

    /// Constructs from the retained selection and sources.
    fn bind(self, mechanisms: &M) -> Result<Self::Execution, Self::Error>;
}

/// Realized transport for a selected tensor group.
pub struct RealtimeTensorResources<B: eredu_core::communication::CommunicationBackend, G, R, I>
where
    B::ParallelContext: Sized,
{
    /// Communication manifest and authority.
    pub communication: crate::PartitionCommunication<B, G, R, I>,
    /// Selected tensor collective.
    pub parallel: B::ParallelContext,
    /// Executor owning the communication submissions.
    pub executor: B::OwnedExecutor,
}

/// Live telemetry of installed parameter-unit residency.
pub trait UnitResidencyTelemetry {
    /// Failure reading the native policy.
    type Error;

    /// Logical residency and transfers.
    fn residency_report(&self) -> Result<crate::ResidencyReport, Self::Error>;

    /// Dense disk-stream counters when active.
    fn dense_stream_report(&self) -> Result<Option<crate::DenseDiskStreamReport>, Self::Error>;

    /// Per-execution-group residency.
    fn execution_group_reports(&self) -> Result<Vec<crate::ResidentLayerGroupReport>, Self::Error>;
}

/// Resident or bounded runtime chosen by the authoritative realization.
pub enum RealtimeLayerwiseRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit>,
{
    /// Fully resident execution.
    Resident(LayerwiseRuntime<A, B, S, R>),
    /// Host-windowed or disk-streamed execution.
    Bounded(LayerwiseRuntime<A, B, S, P>),
}

/// Constructed static execution and mechanisms, detached from mutable model state.
pub struct ConstructedRealtimeExecution<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
{
    selected: SelectedRealtimeRealization,
    execution: RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    backend: PhantomData<fn() -> B>,
}

impl<A, B, M> ConstructedRealtimeExecution<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
{
    /// Returns the immutable selected realization.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    /// Returns the selected resident or bounded runtime.
    pub const fn execution(
        &self,
    ) -> &RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy> {
        &self.execution
    }

    /// Mutably borrows the selected resident or bounded runtime.
    pub fn execution_mut(
        &mut self,
    ) -> &mut RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy> {
        &mut self.execution
    }

    /// Decomposes selected construction for installation in the exact
    /// topology-specific neutral execution runtime.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        SelectedRealtimeRealization,
        RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    ) {
        (self.selected, self.execution)
    }
}

/// Constructs static modules, selected units and their exact parameter storage.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn construct_realtime_execution<A, B, M>(
    mut architecture: A,
    source_architecture: Option<A>,
    prepared: PreparedRealtimeModelContract,
    source: eredu_checkpoint::store::SharedCheckpointSource,
    local_layout: Option<&crate::LocalModelLayout>,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    mechanisms: &M,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    (
        ConstructedRealtimeExecution<A, B, M>,
        crate::LayerwiseModelMetadata,
    ),
    RealtimeModelConstructionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + CheckpointParameterBackend,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State> + RealtimeArchitectureIdentity,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display + From<eredu_core::BackendFailure>,
{
    let (selected, tasks) = prepared.into_parts();
    let requires_source_architecture = selected
        .weight_lowerings()
        .iter()
        .any(|lowering| lowering.kind().transforms_payload());
    if requires_source_architecture != source_architecture.is_some() {
        return Err(RealtimeModelConstructionError::Contract(
            "selected realtime lowering and source-format architecture presence differ".into(),
        ));
    }
    validate_architecture::<A, B, M::State>(&architecture, &selected, false, context)
        .map_err(widen_error)?;
    if let Some(source) = source_architecture.as_ref() {
        validate_architecture::<A, B, M::State>(source, &selected, true, context)
            .map_err(widen_error)?;
    }
    let invocations = architecture
        .execution_unit_storage_map()
        .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
    invocations
        .validate_layout(selected.execution_units())
        .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
    let units = if selected.residency().is_fully_resident() || requires_source_architecture {
        construct_units::<A, B, M::State>(&architecture, &invocations, context)
            .map_err(widen_error)?
    } else {
        Vec::new()
    };
    preflight_realtime_materialization_tasks::<B>(&tasks, source.as_ref())
        .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
    let (source, materialization) = match (requires_source_architecture, transform) {
        (true, Some(quantization)) => {
            let targets = realtime_quantization_targets(&tasks, quantization)
                .map_err(RealtimeModelConstructionError::Contract)?;
            let targets =
                crate::exact_parameter_materialization::bind_packed_transform_targets::<
                    B,
                    M::ParameterEncoding,
                    _,
                    _,
                >(architecture.static_modules(), &units, quantization, targets)
                .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
            let (source, report) =
                crate::exact_parameter_materialization::materialize_exact_quantization::<
                    B,
                    M::ParameterEncoding,
                >(source, quantization, targets, context)
                .map_err(|error| RealtimeModelConstructionError::Mechanism(error.into()))?;
            (source, Some(report))
        }
        (false, None) => (source, None),
        _ => {
            return Err(RealtimeModelConstructionError::Contract(
                "selected realtime lowering differs from the weight transform".into(),
            ))
        }
    };
    let RealtimeTaskBindingPlan {
        mut pinned,
        units: mut unit_bindings,
    } = realtime_task_binding_plan(&tasks, source.as_ref())
        .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
    if let Some(layout) = local_layout {
        let place = |bindings| {
            crate::place_weight_bindings(bindings, source.as_ref(), layout)
                .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))
        };
        pinned = place(pinned)?;
        for bindings in unit_bindings.values_mut() {
            *bindings = place(std::mem::take(bindings))?;
        }
    }
    let layout = invocations.storage_layout();
    let bindings = (0..layout.len())
        .map(|ordinal| {
            let address = layout.address(ordinal).expect("admitted storage ordinal");
            let group = layout
                .group_id(address.group())
                .expect("admitted storage group");
            unit_bindings
                .remove(&ParameterGroupOwner::execution_unit(
                    group.clone(),
                    address.index(),
                ))
                .ok_or_else(|| {
                    RealtimeModelConstructionError::Contract(format!(
                        "selected unit bindings are missing for {address:?}"
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !unit_bindings.is_empty() {
        return Err(RealtimeModelConstructionError::Contract(
            "selected realtime task bindings contain unowned units".into(),
        ));
    }

    crate::checkpoint_quantization::realize_prepared_quantization(&source, || {
        mechanisms.unit_quantizer()
    })
    .map_err(RealtimeModelConstructionError::Storage)?;
    let resources = crate::prepare_parameter_resources(
        source.as_ref(),
        &invocations,
        selected.residency(),
        pinned,
        bindings,
        Vec::new(),
        |_| false,
        |bindings| mechanisms.unit_host_capacity(bindings),
    )
    .map_err(|error| RealtimeModelConstructionError::Contract(error.to_string()))?;
    let (storage, mut metadata) = crate::parameter_units::materialize_parameter_units(
        mechanisms,
        source,
        architecture.static_modules_mut(),
        resources,
        None,
        None,
        None,
    )
    .map_err(RealtimeModelConstructionError::Storage)?;
    metadata.set_materialization(materialization);

    let execution = if selected.residency().is_fully_resident() {
        let policy = <M::ResidentPolicy as crate::bounded_execution::units::BindResidentPolicy<
            B,
            A::Unit,
            crate::ParameterUnitStorage<B, M>,
        >>::bind_resident(storage, units, context)
        .map_err(|error| RealtimeModelConstructionError::Mechanism(error.into()))?;
        RealtimeLayerwiseRuntime::Resident(LayerwiseRuntime::new(architecture, policy))
    } else {
        let policy = <M::BoundedPolicy as crate::bounded_execution::units::BindBoundedPolicy<
            A::Unit,
            crate::ParameterUnitStorage<B, M>,
        >>::bind_bounded(storage);
        RealtimeLayerwiseRuntime::Bounded(LayerwiseRuntime::new(architecture, policy))
    };
    Ok((
        ConstructedRealtimeExecution {
            selected,
            execution,
            backend: PhantomData,
        },
        metadata,
    ))
}

fn widen_error<A, P, M>(
    error: RealtimeModelConstructionError<A, std::convert::Infallible, std::convert::Infallible>,
) -> RealtimeModelConstructionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    match error {
        RealtimeModelConstructionError::Architecture(error) => {
            RealtimeModelConstructionError::Architecture(error)
        }
        RealtimeModelConstructionError::Contract(error) => {
            RealtimeModelConstructionError::Contract(error)
        }
        RealtimeModelConstructionError::Mechanism(error) => match error {},
        RealtimeModelConstructionError::Policy(error) => match error {},
        RealtimeModelConstructionError::Storage(error) => {
            RealtimeModelConstructionError::Storage(error)
        }
    }
}

fn construct_units<A, B, S>(
    architecture: &A,
    invocations: &crate::ExecutionUnitAddressMap,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    Vec<A::Unit>,
    RealtimeModelConstructionError<A::Error, std::convert::Infallible, std::convert::Infallible>,
>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    (0..invocations.storage_layout().len())
        .map(|ordinal| {
            let storage = invocations
                .storage_location(ordinal)
                .expect("admitted physical storage ordinal");
            architecture
                .build_storage_unit(storage, context)
                .map_err(RealtimeModelConstructionError::Architecture)
        })
        .collect()
}

fn validate_architecture<A, B, S>(
    architecture: &A,
    selected: &SelectedRealtimeRealization,
    source: bool,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (),
    RealtimeModelConstructionError<A::Error, std::convert::Infallible, std::convert::Infallible>,
>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + RealtimeArchitectureIdentity,
    A::Error: std::fmt::Display,
{
    if !source {
        let actual = architecture
            .realtime_construction_identity()
            .map_err(RealtimeModelConstructionError::Contract)?;
        let requirements = selected.requirements();
        let expected = RealtimeArchitectureConstructionIdentity::new(
            requirements.architecture().clone(),
            requirements.frame_contract().policy_identity().clone(),
            requirements.state_layout_identity().clone(),
        );
        if actual != expected {
            return Err(RealtimeModelConstructionError::Contract(
                "constructed realtime architecture identities differ from selection".into(),
            ));
        }
    }
    if architecture
        .execution_graph()
        .map_err(RealtimeModelConstructionError::Architecture)?
        != *selected.execution_graph()
    {
        return Err(RealtimeModelConstructionError::Contract(
            "constructed realtime execution graph differs from selection".into(),
        ));
    }
    for group in 0..selected.execution_graph().groups().len() {
        let actual = architecture
            .group_unit_count(group)
            .map_err(RealtimeModelConstructionError::Architecture)?;
        let expected = selected
            .execution_units()
            .group_range(group)
            .expect("selected layout contains every graph group")
            .len();
        if actual != expected {
            return Err(RealtimeModelConstructionError::Contract(format!(
                "constructed realtime group {group} unit count differs from selection"
            )));
        }
    }
    let actual = architecture
        .parameter_description(context)
        .map_err(RealtimeModelConstructionError::Architecture)?;
    let expected = if source {
        selected.source_parameters()
    } else {
        selected.execution_parameters()
    };
    if &actual != expected {
        return Err(RealtimeModelConstructionError::Contract(
            "constructed realtime parameter topology differs from selection".into(),
        ));
    }
    if !source
        && architecture
            .state_layout()
            .map_err(RealtimeModelConstructionError::Architecture)?
            != *selected.state().layout()
    {
        return Err(RealtimeModelConstructionError::Contract(
            "constructed realtime state layout differs from selection".into(),
        ));
    }
    Ok(())
}

/// Invalid selected recipe or task handoff.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeModelContractError {
    /// A component's retained physical sources differ from its selected recipe traversal.
    #[error("realtime source provenance differs for target {target}")]
    SourceProvenanceMismatch {
        /// Affected selected component target.
        target: String,
    },
    /// A source-backed companion's inferred geometry differs from selection.
    #[error("realtime recipe geometry differs for target {target}")]
    RecipeGeometryMismatch {
        /// Affected selected component target.
        target: String,
    },
    /// Primary recipe scalar encoding differs from its selected source descriptor.
    #[error("realtime recipe encoding differs for target {target}")]
    RecipeEncodingMismatch {
        /// Affected selected component target.
        target: String,
    },
    /// Concrete component payload does not exactly cover the lowering.
    #[error("realtime component payload differs for target {target}")]
    ComponentCoverageMismatch {
        /// Affected selected lowering target.
        target: String,
    },
    /// Task targets do not exactly cover selection.
    #[error("realtime materialization tasks do not exactly cover selected targets")]
    TaskCoverageMismatch,
    /// A task contains a lowering that differs from selection.
    #[error("realtime materialization task differs from selection for target {target}")]
    TaskSelectionMismatch {
        /// Affected selected lowering target.
        target: String,
    },
    /// A selected target has no declared execution parameter owner.
    #[error("realtime materialization target {target} has no execution owner")]
    TaskOwnerUnavailable {
        /// Affected selected lowering target.
        target: String,
    },
    /// A task owner differs from the exact selected execution owner.
    #[error("realtime materialization owner differs from selection for target {target}")]
    TaskOwnerMismatch {
        /// Affected selected lowering target.
        target: String,
    },
    /// Canonical task-to-binding derivation or metadata preflight failed.
    #[error("realtime binding plan is invalid: {detail}")]
    BindingPlan {
        /// Exact metadata-only causal failure.
        detail: String,
    },
}

/// Failure while constructing one selected layered realtime model.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeModelConstructionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    /// Neutral architecture construction failed.
    #[error("realtime architecture construction failed: {0}")]
    Architecture(A),
    /// Selected and constructed neutral contracts differ.
    #[error("invalid realtime model construction: {0}")]
    Contract(String),
    /// Generic materialization, state, or storage mechanism failed.
    #[error("realtime model mechanism failed: {0}")]
    Mechanism(M),
    /// Native parameter storage preparation failed.
    #[error("realtime parameter storage failed: {0}")]
    Storage(#[source] eredu_core::BackendFailure),
    /// Resident/bounded policy failure reserved for execution-time composition.
    #[error("realtime residency policy failed: {0}")]
    Policy(P),
}
