//! Family-blind construction of a selected layered realtime model.

use std::marker::PhantomData;

use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeMetadata};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    LayerWeightResidency, LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime,
    ParameterGroupOwner, RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, RuntimeState, SelectedRealtimeRealization,
    SubmissionBackend,
};

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

/// Concrete state paired with the exact realization its mechanism implemented.
pub struct RealizedRealtimeState<S> {
    state: S,
    realization: crate::SelectedRealtimeStateRealization,
}

impl<S> RealizedRealtimeState<S> {
    /// Attaches an exact state realization report to concrete storage.
    pub fn new(state: S, realization: crate::SelectedRealtimeStateRealization) -> Self {
        Self { state, realization }
    }

    /// Consumes concrete state and its exact realization report.
    pub fn into_parts(self) -> (S, crate::SelectedRealtimeStateRealization) {
        (self.state, self.realization)
    }
}

/// Concrete layer storage paired with the exact residency it implemented.
pub struct RealizedRealtimePolicy<P> {
    policy: P,
    residency: LayerWeightResidency,
}

impl<P> RealizedRealtimePolicy<P> {
    /// Attaches the implemented residency report to a concrete policy.
    pub fn new(policy: P, residency: LayerWeightResidency) -> Self {
        Self { policy, residency }
    }

    /// Consumes concrete storage and its exact residency report.
    pub fn into_parts(self) -> (P, LayerWeightResidency) {
        (self.policy, self.residency)
    }
}

/// One selected component paired with its exact architecture recipe payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RealtimeMaterializationComponent {
    requirement: RealtimeWeightComponentRequirement,
}

impl RealtimeMaterializationComponent {
    /// Pairs an exact selected component with recipe data, when source-backed.
    pub const fn new(requirement: RealtimeWeightComponentRequirement) -> Self {
        Self { requirement }
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

/// Selected realization paired with complete architecture recipe payloads.
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
pub trait RealtimeModelConstructionMechanisms<A, B>
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
    type ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Host-windowed or disk-streamed unit storage.
    type BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Mechanism failure.
    type Error;

    /// Materializes and binds every resident static/unit component exactly once.
    #[allow(clippy::too_many_arguments)]
    fn prepare_resident_materialization(
        &mut self,
        architecture: &mut A,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Builds fully resident storage around populated units.
    fn resident_policy(
        &mut self,
        architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedRealtimeRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<RealizedRealtimePolicy<Self::ResidentPolicy>, Self::Error>;

    /// Builds selected bounded storage without eagerly constructing every unit.
    #[allow(clippy::too_many_arguments)]
    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        source_architecture: Option<&mut A>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<RealizedRealtimePolicy<Self::BoundedPolicy>, Self::Error>;

    /// Realizes exact selected state components and placements.
    fn realize_state(
        &mut self,
        selected: &crate::SelectedRealtimeStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<RealizedRealtimeState<Self::State>, Self::Error>;
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
    mechanisms: M,
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

    /// Returns generic construction mechanisms for reports or later composition.
    pub const fn mechanisms(&self) -> &M {
        &self.mechanisms
    }

    /// Mutably borrows generic construction mechanisms.
    pub fn mechanisms_mut(&mut self) -> &mut M {
        &mut self.mechanisms
    }

    /// Decomposes selected construction for installation in the exact
    /// topology-specific neutral execution runtime.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        SelectedRealtimeRealization,
        RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
        M,
    ) {
        (self.selected, self.execution, self.mechanisms)
    }
}

/// Fully constructed execution paired with its initial mutable model state.
pub struct ConstructedRealtimeModel<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
{
    execution: ConstructedRealtimeExecution<A, B, M>,
    state: M::State,
}

impl<A, B, M> ConstructedRealtimeModel<A, B, M>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
{
    /// Returns the immutable selected realization.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        self.execution.selected()
    }

    /// Returns the selected resident or bounded runtime.
    pub const fn execution(
        &self,
    ) -> &RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy> {
        self.execution.execution()
    }

    /// Mutably borrows the selected resident or bounded runtime.
    pub fn execution_mut(
        &mut self,
    ) -> &mut RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy> {
        self.execution.execution_mut()
    }

    /// Returns realized architecture state.
    pub const fn state(&self) -> &M::State {
        &self.state
    }

    /// Mutably borrows realized architecture state.
    pub fn state_mut(&mut self) -> &mut M::State {
        &mut self.state
    }

    /// Mutably borrows execution and state together for one atomic model pass.
    #[allow(clippy::type_complexity)]
    pub fn execution_and_state_mut(
        &mut self,
    ) -> (
        &mut RealtimeLayerwiseRuntime<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
        &mut M::State,
    ) {
        (self.execution.execution_mut(), &mut self.state)
    }

    /// Mutably borrows detached execution and state as separate typed values.
    #[allow(clippy::type_complexity)]
    pub fn constructed_execution_and_state_mut(
        &mut self,
    ) -> (&mut ConstructedRealtimeExecution<A, B, M>, &mut M::State) {
        (&mut self.execution, &mut self.state)
    }

    /// Returns generic construction mechanisms for reports or later composition.
    pub const fn mechanisms(&self) -> &M {
        self.execution.mechanisms()
    }

    /// Mutably borrows generic mechanisms.
    pub fn mechanisms_mut(&mut self) -> &mut M {
        self.execution.mechanisms_mut()
    }

    /// Consumes the combined model into static execution and mutable state.
    pub fn into_execution_and_state(self) -> (ConstructedRealtimeExecution<A, B, M>, M::State) {
        (self.execution, self.state)
    }
}

/// Constructs static modules, selected units, storage, and exact state once.
#[allow(clippy::type_complexity)]
pub fn construct_realtime_model<A, B, M>(
    mut architecture: A,
    mut source_architecture: Option<A>,
    prepared: PreparedRealtimeModelContract,
    mut mechanisms: M,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    ConstructedRealtimeModel<A, B, M>,
    RealtimeModelConstructionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: RealtimeModelConstructionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State> + RealtimeArchitectureIdentity,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let (selected, tasks) = prepared.into_parts();
    let requires_source_architecture = selected.weight_lowerings().iter().any(|lowering| {
        matches!(
            lowering.kind(),
            crate::WeightLoweringKind::Transform | crate::WeightLoweringKind::DerivedTransform
        )
    });
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
    let (state, state_realization) = mechanisms
        .realize_state(selected.state(), context)
        .map_err(RealtimeModelConstructionError::Mechanism)?
        .into_parts();
    if &state_realization != selected.state() || state.layout() != selected.state().layout() {
        return Err(RealtimeModelConstructionError::Contract(
            "realized realtime state differs from selection".into(),
        ));
    }
    let execution = match selected.residency() {
        LayerWeightResidency::FullyResident => {
            let mut units = construct_units::<A, B, M::State>(&architecture, &selected, context)
                .map_err(widen_error)?;
            let mut source_units = source_architecture
                .as_ref()
                .map(|source| construct_units::<A, B, M::State>(source, &selected, context))
                .transpose()
                .map_err(widen_error)?;
            mechanisms
                .prepare_resident_materialization(
                    &mut architecture,
                    &mut units,
                    source_architecture.as_mut(),
                    source_units.as_deref_mut(),
                    &tasks,
                    &selected,
                    context,
                )
                .map_err(RealtimeModelConstructionError::Mechanism)?;
            let (policy, residency) = mechanisms
                .resident_policy(&mut architecture, units, &selected, context)
                .map_err(RealtimeModelConstructionError::Mechanism)?
                .into_parts();
            if residency != selected.residency() {
                return Err(RealtimeModelConstructionError::Contract(
                    "realized realtime weight residency differs from selection".into(),
                ));
            }
            RealtimeLayerwiseRuntime::Resident(LayerwiseRuntime::new(architecture, policy))
        }
        LayerWeightResidency::LayerwiseHost(_) | LayerWeightResidency::DenseDiskStream(_) => {
            let (policy, residency) = mechanisms
                .bounded_policy(
                    &mut architecture,
                    source_architecture.as_mut(),
                    &tasks,
                    &selected,
                    context,
                )
                .map_err(RealtimeModelConstructionError::Mechanism)?
                .into_parts();
            if residency != selected.residency() {
                return Err(RealtimeModelConstructionError::Contract(
                    "realized realtime weight residency differs from selection".into(),
                ));
            }
            RealtimeLayerwiseRuntime::Bounded(LayerwiseRuntime::new(architecture, policy))
        }
    };
    Ok(ConstructedRealtimeModel {
        execution: ConstructedRealtimeExecution {
            selected,
            execution,
            mechanisms,
            backend: PhantomData,
        },
        state,
    })
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
    }
}

fn construct_units<A, B, S>(
    architecture: &A,
    selected: &SelectedRealtimeRealization,
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
    (0..selected.execution_units().len())
        .map(|ordinal| {
            let address = selected
                .execution_units()
                .address(ordinal)
                .expect("selected execution ordinal has an address");
            architecture
                .build_unit(address.group(), address.index(), context)
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
            requirements.speech_schedule_identity().clone(),
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
    /// Resident/bounded policy failure reserved for execution-time composition.
    #[error("realtime residency policy failed: {0}")]
    Policy(P),
}
