//! Validated component declarations and typed equation adapters.

use crate::{
    ArchitectureGroupTransport, ArchitectureParameterDescription, ArchitectureParameters,
    ArchitectureStatePartitionPlan, ExecutionGraph, ExecutionUnitAddress, LayeredArchitecture,
    LayeredForwardState, ModelStateIdentity, ParallelLayeredArchitecture, PartitionState,
    RuntimeState, StateError, StateLayout, StaticParameterVisitor, StaticParameterVisitorMut,
};
use eredu_nn::{NeuralBackend, Parameterized, Tensor};
use std::{borrow::Borrow, ops::Range};

/// Invalid component declaration or execution address.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ComponentDefinitionError {
    /// State partition rules disagree with the declared graph or geometry.
    #[error("component {component:?}: {source}")]
    StatePartition {
        /// Stable component identity.
        component: String,
        /// Exact invalid state contract.
        #[source]
        source: crate::ArchitectureStatePartitionError,
    },
    /// A declaration does not cover its canonical resources.
    #[error("component {component:?}: invalid {resource}")]
    Invalid {
        /// Stable component identity.
        component: String,
        /// Invalid resource or contract.
        resource: String,
    },
    /// A caller supplied an address outside the retained declaration.
    #[error("component {component:?}: unknown unit {group}:{index}")]
    Address {
        /// Stable component identity.
        component: String,
        /// Requested group.
        group: usize,
        /// Requested unit.
        index: usize,
    },
}

/// Stable identity of canonical parameter storage, distinct from an invocation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParameterStorageId(pub crate::ParameterGroupOwner);

/// Architecture-global identity of one mutable state slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateSlotId(pub usize);

/// One logical invocation and its independent state references.
#[derive(Debug, Clone)]
pub struct ComponentUnit {
    /// Stable architecture-owned operation path.
    pub path: String,
    /// Contiguous mutable-state slots retained through completion.
    pub state: Range<usize>,
}

/// One retained semantic definition used by shared execution contracts.
///
/// Parameter descriptions supply the validated graph and storage ownership;
/// paths and state references follow its canonical invocation layout. Group
/// transport and primary/prediction identities are retained with the definition.
#[derive(Debug, Clone)]
pub struct ComponentDefinition {
    id: String,
    parameters: ArchitectureParameterDescription,
    units: Vec<ComponentUnit>,
    transport: Vec<ArchitectureGroupTransport>,
    primary: String,
    prediction: Vec<String>,
    state: StateLayout,
    state_partition: ArchitectureStatePartitionPlan,
}

impl ComponentDefinition {
    /// Validates every declared invocation and group reference without native resources.
    pub fn new(
        id: impl Into<String>,
        parameters: ArchitectureParameterDescription,
        units: Vec<ComponentUnit>,
        transport: Vec<ArchitectureGroupTransport>,
        primary: impl Into<String>,
        prediction: Vec<String>,
        state: StateLayout,
        state_partition: ArchitectureStatePartitionPlan,
    ) -> Result<Self, ComponentDefinitionError> {
        let id = id.into();
        let primary = primary.into();
        let invalid = |resource: &str| ComponentDefinitionError::Invalid {
            component: id.clone(),
            resource: resource.into(),
        };
        if id.trim().is_empty() {
            return Err(invalid("identity"));
        }
        let graph = parameters.graph();
        if graph.group_index(&primary).is_none() {
            return Err(invalid("primary group"));
        }
        let mut seen = std::collections::BTreeSet::new();
        for group in &prediction {
            if graph.group_index(group).is_none() || !seen.insert(group) {
                return Err(invalid("prediction group"));
            }
        }
        if units.len() != parameters.unit_layout().len() {
            return Err(invalid("unit count"));
        }
        if transport.len() != graph.groups().len() {
            return Err(invalid("group transport count"));
        }
        let mut paths = std::collections::BTreeSet::new();
        for unit in &units {
            if unit.path.trim().is_empty() || !paths.insert(&unit.path) {
                return Err(invalid(&format!("unit path {:?}", unit.path)));
            }
            if unit.state.start > unit.state.end || unit.state.end > state.len() {
                return Err(invalid(&format!("state mapping for {:?}", unit.path)));
            }
        }
        for (group, contract) in transport.iter().enumerate() {
            if contract.request_optional
                && (!matches!(
                    contract.kind,
                    crate::ArchitectureGroupKind::VisionEncoder
                        | crate::ArchitectureGroupKind::AudioEncoder
                ) || !graph.groups()[group].dependencies().is_empty())
            {
                return Err(invalid(&format!("optional group {group}")));
            }
        }
        state_partition
            .resolve_partition(
                &state,
                parameters.unit_layout(),
                |group| {
                    parameters
                        .unit_layout()
                        .group_range(group)
                        .map(|range| 0..range.len())
                },
                true,
            )
            .map_err(|source| ComponentDefinitionError::StatePartition {
                component: id.clone(),
                source,
            })?;
        Ok(Self {
            id,
            parameters,
            units,
            transport,
            primary,
            prediction,
            state,
            state_partition,
        })
    }
    /// Canonical storage and graph ownership.
    pub fn parameters(&self) -> &ArchitectureParameterDescription {
        &self.parameters
    }
    /// Mutable-state geometry.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state
    }
    /// Resolves a logical invocation using the canonical address map.
    pub fn unit(
        &self,
        group: usize,
        index: usize,
    ) -> Result<&ComponentUnit, ComponentDefinitionError> {
        self.parameters
            .unit_layout()
            .ordinal(group, index)
            .map(|ordinal| &self.units[ordinal])
            .ok_or_else(|| ComponentDefinitionError::Address {
                component: self.id.clone(),
                group,
                index,
            })
    }
    /// Canonical parameter owners consumed by an invocation. Storage sharing is
    /// independent of the invocation's mutable-state references.
    pub fn parameter_storage(
        &self,
        group: usize,
        index: usize,
    ) -> Result<Vec<ParameterStorageId>, ComponentDefinitionError> {
        self.unit(group, index)?;
        let group_id = self
            .parameters
            .unit_layout()
            .group_id(group)
            .expect("validated group");
        Ok(self
            .parameters
            .groups()
            .iter()
            .filter(|owned| match owned.owner() {
                crate::ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                } => group == group_id && *global_unit == index,
                crate::ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => consumers
                    .iter()
                    .any(|(group, unit)| group == group_id && *unit == index),
                _ => false,
            })
            .map(|owned| ParameterStorageId(owned.owner().clone()))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect())
    }
    /// Mutable-state identities retained by this invocation.
    pub fn state_slots(
        &self,
        group: usize,
        index: usize,
    ) -> Result<impl Iterator<Item = StateSlotId>, ComponentDefinitionError> {
        Ok(self.unit(group, index)?.state.clone().map(StateSlotId))
    }
    /// Transfers the canonical cold parameter projection.
    pub fn into_parameters(self) -> ArchitectureParameterDescription {
        self.parameters
    }
}

/// Shared pinned-module representation for components with only leased resources.
#[derive(Debug, Default)]
pub struct EmptyParameterModule;
impl<T: 'static> Parameterized<T> for EmptyParameterModule {
    fn visit_parameters<'a, V>(&'a self, _: &mut V)
    where
        V: eredu_nn::ParameterVisitor<'a, T>,
    {
    }
    fn visit_parameters_mut<'a, V>(&'a mut self, _: &mut V)
    where
        V: eredu_nn::ParameterVisitorMut<'a, T>,
    {
    }
    fn set_trainable(&mut self, _: bool) {}
}

/// Shared host-only traversal state for stateless components.
/// The state driver requires one `NoState` layout entry; it owns no tensor.
pub struct StatelessComponentState {
    layout: StateLayout,
}
impl StatelessComponentState {
    /// Constructs the neutral driver metadata without allocating mutable state.
    pub fn new() -> Result<Self, StateError> {
        Ok(Self {
            layout: Self::state_layout()?,
        })
    }
    /// Returns the common driver layout for a component without mutable state.
    pub fn state_layout() -> Result<StateLayout, StateError> {
        let schedule =
            eredu_core::LayerSchedule::new(1, vec![eredu_core::cache::LayerCachePolicy::NoState])
                .expect("one stateless entry");
        StateLayout::new(schedule)
    }
    /// Borrows the retained neutral metadata.
    pub fn layout(&self) -> &StateLayout {
        &self.layout
    }
}
impl<B: NeuralBackend> RuntimeState<B> for StatelessComponentState {
    type RetainedValues<'a>
        = std::iter::Empty<&'a B::Tensor>
    where
        B::Tensor: 'a;
    fn layout(&self) -> &StateLayout {
        &self.layout
    }
    fn optional_layout(&self) -> Option<&StateLayout> {
        None
    }
    fn retained_values(
        &self,
        ordinal: usize,
        _: ExecutionUnitAddress,
    ) -> Result<Self::RetainedValues<'_>, StateError> {
        if ordinal != 0 {
            return Err(StateError::UnknownLayer {
                layer: ordinal,
                count: 1,
            });
        }
        Ok(std::iter::empty())
    }
}

/// Family equations for a component with independently leased parameter modules.
/// Implementing this trait alone declares no parallel or partition capability.
pub trait ComponentKernel<B: NeuralBackend> {
    /// Mutable state selected by the family, independent of parameter storage.
    type State: RuntimeState<B>;
    /// Borrowed typed operation.
    type Input<'a>
    where
        Self: 'a;
    /// One unloaded parameter resource.
    type Unit: Parameterized<B::Tensor>;
    /// Request-local values retained through ordinary completion.
    type ForwardContext;
    /// Allocation-free transient resource roots.
    type RetainedContextValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    /// Family failure, preserving native sources through neutral wrappers.
    type Error: std::error::Error + Send + Sync + From<ComponentDefinitionError> + 'static;
    /// Family cache identity for the selected realization.
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Self::Error>;
    /// Builds one unloaded resource after shared address validation.
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error>;
    /// Prepares a typed operation.
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut Self::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Selects or assembles group input after shared dependency-count validation.
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut Self::State,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Operation-specific group activity.
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool;
    /// Runs the family equation against an already admitted resource.
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut Self::State,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Converts the final activation to the typed operation's ordinary tensor result.
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut Self::State,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Retains operation inputs and candidates until native completion is established.
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a>;
}

/// Explicit parallel equations; availability is enforced by a trait bound.
pub trait ParallelComponentKernel<B: NeuralBackend>: ComponentKernel<B> {
    /// Parallel ingress and communicator validation.
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut Self::State,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Parallel unit equation, including architecture-specific collectives.
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut Self::State,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Parallel readout.
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut Self::State,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Shared driver-facing contracts over one declaration and typed family kernel.
pub struct ComponentAdapter<D, K> {
    definition: D,
    kernel: K,
    empty: EmptyParameterModule,
}
impl<D: Borrow<ComponentDefinition>, K> ComponentAdapter<D, K> {
    /// Couples a validated declaration with its family equations.
    pub fn new(definition: D, kernel: K) -> Result<Self, ComponentDefinitionError> {
        let declaration = definition.borrow();
        if declaration.parameters.groups().iter().any(|group| {
            !matches!(
                group.owner(),
                crate::ParameterGroupOwner::ExecutionUnit { .. }
            )
        }) {
            return Err(ComponentDefinitionError::Invalid {
                component: declaration.id.clone(),
                resource: "leased component adapter contains pinned parameters".into(),
            });
        }
        Ok(Self {
            definition,
            kernel,
            empty: EmptyParameterModule,
        })
    }
    /// Borrows family semantic operations and selected geometry.
    pub fn kernel(&self) -> &K {
        &self.kernel
    }
    /// Mutably borrows family semantic operations.
    pub fn kernel_mut(&mut self) -> &mut K {
        &mut self.kernel
    }
    /// Borrows the one retained definition.
    pub fn definition(&self) -> &ComponentDefinition {
        self.definition.borrow()
    }
    fn address(
        &self,
        group: usize,
        index: usize,
    ) -> Result<ExecutionUnitAddress, ComponentDefinitionError> {
        let definition = self.definition();
        definition.unit(group, index)?;
        let layout = definition.parameters.unit_layout();
        Ok(layout
            .address(layout.ordinal(group, index).expect("validated unit"))
            .expect("validated ordinal"))
    }
}
impl<B, D, K> ArchitectureParameters<B> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    K: ComponentKernel<B> + 'static,
{
    type DefinitionError = K::Error;
    fn state_layout(&self) -> Result<StateLayout, K::Error> {
        Ok(self.definition().state.clone())
    }
    fn state_identity(
        &self,
        state: &PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<ModelStateIdentity, K::Error> {
        self.kernel.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, K::Error> {
        Ok(self.definition().parameters.clone())
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}
impl<B, D, K> LayeredArchitecture<B, K::State> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    K: ComponentKernel<B> + 'static,
{
    type Input<'a>
        = K::Input<'a>
    where
        Self: 'a;
    type StaticModules = EmptyParameterModule;
    type Unit = K::Unit;
    type ForwardContext = K::ForwardContext;
    type RetainedContextValues<'a>
        = K::RetainedContextValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = K::Error;
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.definition().transport[group].clone()
    }
    fn primary_execution_group(&self) -> &str {
        &self.definition().primary
    }
    fn prediction_execution_groups(&self) -> Vec<String> {
        self.definition().prediction.clone()
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        self.definition().state_partition.clone()
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, K::Error> {
        Ok(self.definition().parameters.graph().clone())
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, K::Error> {
        self.definition()
            .parameters
            .unit_layout()
            .group_range(group)
            .map(|range| range.len())
            .ok_or_else(|| {
                ComponentDefinitionError::Address {
                    component: self.definition().id.clone(),
                    group,
                    index: 0,
                }
                .into()
            })
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, K::Error> {
        Ok(self.definition().unit(group, index)?.path.clone())
    }
    fn static_modules(&self) -> &EmptyParameterModule {
        &self.empty
    }
    fn static_modules_mut(&mut self) -> &mut EmptyParameterModule {
        &mut self.empty
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<K::Unit, K::Error> {
        self.kernel.build_unit(self.address(group, index)?, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut K::State,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        self.kernel.begin(input, state, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut K::State,
        forward: &mut K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        if self
            .definition()
            .parameters
            .graph()
            .dependencies(group)
            .is_none_or(|declared| declared.len() != dependencies.len())
        {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: format!("dependencies for group {group}"),
            }
            .into());
        }
        self.kernel
            .begin_group(group, initial, dependencies, state, forward, context)
    }
    fn should_execute_group(&self, group: usize, forward: &K::ForwardContext) -> bool {
        self.kernel.executes_group(group, forward)
    }
    fn retained_state_ordinals(&self, group: usize, index: usize, _: usize) -> Range<usize> {
        self.definition()
            .unit(group, index)
            .expect("validated traversal address")
            .state
            .clone()
    }
    fn state_ordinal(&self, group: usize, index: usize, _: usize) -> usize {
        self.definition()
            .unit(group, index)
            .expect("validated traversal address")
            .state
            .start
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut K::Unit,
        hidden: &B::Tensor,
        state: &mut K::State,
        forward: &mut K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel.execute(
            self.address(group, index)?,
            unit,
            hidden,
            state,
            forward,
            context,
        )
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut K::State,
        forward: &K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel.finish(hidden, state, forward, context)
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a K::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a> {
        self.kernel.retained_values(
            forward,
            self.address(group, index)
                .expect("validated traversal address"),
        )
    }
}
impl<B, D, K> ParallelLayeredArchitecture<B, K::State> for ComponentAdapter<D, K>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    K: ParallelComponentKernel<B> + 'static,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut K::State,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        self.kernel.begin_parallel(input, state, parallel, context)
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut K::Unit,
        hidden: &B::Tensor,
        state: &mut K::State,
        forward: &mut K::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel.execute_parallel(
            self.address(group, index)?,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut K::State,
        forward: &K::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel
            .finish_parallel(hidden, state, forward, parallel, context)
    }
}
