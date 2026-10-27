//! Validated component declarations and typed equation adapters.

mod parameter_paths;
mod projection;

use crate::{
    ActivationObserver, ArchitectureGroupTransport, ArchitectureParameterDescription,
    ArchitectureParameters, ArchitectureStatePartitionPlan, ComponentResourceAccess,
    ExecutionGraph, ExecutionUnitAddress, LayeredArchitecture, LayeredForwardState,
    ModelStateIdentity, ParallelLayeredArchitecture, PartitionState, RuntimeState, StateError,
    StateLayout, StaticParameterVisitor, StaticParameterVisitorMut,
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
    /// A projected parameter declaration violates canonical ownership or placement.
    #[error("component {component:?}: {source}")]
    Parameters {
        /// Stable component identity.
        component: String,
        /// Exact invalid parameter contract.
        #[source]
        source: crate::ArchitectureParameterError,
    },
    /// A projected invocation layout violates stable group addressing.
    #[error("component {component:?}: {source}")]
    UnitLayout {
        /// Stable component identity.
        component: String,
        /// Exact invalid invocation layout.
        #[source]
        source: crate::ExecutionUnitLayoutError,
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

impl From<ComponentDefinitionError> for eredu_nn::Error {
    fn from(error: ComponentDefinitionError) -> Self {
        Self::backend_source(error)
    }
}

/// Stable identity of canonical parameter storage, distinct from an invocation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParameterStorageId {
    /// One pinned storage owner, independent of its logical consumers.
    StaticRole(String),
    /// Default independently leased storage for one canonical unit.
    ExecutionUnit {
        /// Stable execution-group identity.
        group: crate::ExecutionGroupId,
        /// Architecture-global unit within the group.
        global_unit: usize,
    },
}

impl ParameterStorageId {
    /// Projects storage identity without including logical-consumer selection.
    pub(crate) fn from_owner(owner: &crate::ParameterGroupOwner) -> Self {
        match owner {
            crate::ParameterGroupOwner::StaticRole(role)
            | crate::ParameterGroupOwner::StaticUnitConsumers { role, .. } => {
                Self::StaticRole(role.clone())
            }
            crate::ParameterGroupOwner::StaticAnyOf(roles) => Self::StaticRole(
                roles
                    .first()
                    .expect("validated shared static owner")
                    .clone(),
            ),
            crate::ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } => Self::ExecutionUnit {
                group: group.clone(),
                global_unit: *global_unit,
            },
        }
    }
}

/// Architecture-global identity of one mutable state slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StateSlotId(pub usize);

/// One logical invocation and its independent state references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentUnit {
    /// Stable architecture-owned operation path.
    pub path: String,
    /// Contiguous mutable-state slots retained through completion.
    pub state: Range<usize>,
}

/// One logical parameter reference at its exact invocation, independent of storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentParameterUse {
    /// Invocation executing the parameterized equation.
    pub invocation: ExecutionUnitAddress,
    /// Canonical parameter target declared by the physical resource owner.
    pub target: String,
}

/// One retained semantic definition used by shared execution contracts.
///
/// Parameter descriptions supply the validated graph and storage ownership;
/// paths and state references follow its canonical invocation layout. Group
/// transport and primary/prediction identities are retained with the definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentDefinition {
    id: String,
    parameters: ArchitectureParameterDescription,
    units: Vec<ComponentUnit>,
    unit_observations: Vec<crate::UnitObservationBoundaries>,
    parameter_uses: std::collections::BTreeMap<String, ComponentParameterUse>,
    transport: Vec<ArchitectureGroupTransport>,
    primary: String,
    prediction: Vec<String>,
    state: StateLayout,
    state_partition: ArchitectureStatePartitionPlan,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParameterOwnershipProjection {
    Exact,
    Subset,
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
            .validate(&state, parameters.unit_layout())
            .map_err(|source| ComponentDefinitionError::StatePartition {
                component: id.clone(),
                source,
            })?;
        for rule in state_partition.rules() {
            if let crate::ArchitectureStatePlacement::Invocation { address } = rule.placement() {
                let ordinal = parameters
                    .unit_layout()
                    .ordinal(address.group(), address.index())
                    .expect("validated state owner");
                let owner = &units[ordinal];
                let slots = rule.layers();
                if owner.state.start > slots.start || owner.state.end < slots.end {
                    return Err(invalid(&format!(
                        "state owner {:?} does not reference slots {slots:?}",
                        owner.path
                    )));
                }
            }
        }
        let unit_observations = units
            .iter()
            .map(|unit| crate::UnitObservationBoundaries::around(&unit.path))
            .collect();
        Ok(Self {
            id,
            parameters,
            units,
            unit_observations,
            parameter_uses: Default::default(),
            transport,
            primary,
            prediction,
            state,
            state_partition,
        })
    }
    /// Stable identity used in typed declaration and binding errors.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Canonical storage and graph ownership.
    pub fn parameters(&self) -> &ArchitectureParameterDescription {
        &self.parameters
    }
    /// Mutable-state geometry.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state
    }
    /// Selects architecture-authored local state geometry on a retained definition.
    /// Identical geometry reuses the declaration; changed geometry consumes a unique
    /// declaration or copies a shared one before validating slot identities and lifetimes.
    pub fn select_state_layout(
        self: std::sync::Arc<Self>,
        state: &StateLayout,
    ) -> Result<std::sync::Arc<Self>, ComponentDefinitionError> {
        if self.state_layout() == state {
            return Ok(self);
        }
        std::sync::Arc::unwrap_or_clone(self)
            .with_state_layout(state.clone())
            .map(std::sync::Arc::new)
    }

    /// Retains invocation and state-slot identities with architecture-selected local geometry.
    /// Segment lifetimes, component roles, dtypes and residency stay unchanged.
    pub fn with_state_layout(self, state: StateLayout) -> Result<Self, ComponentDefinitionError> {
        self.project_state_layout(state, &std::collections::BTreeSet::new())
    }

    /// Declares persistent resources for selected dormant slots, such as receiver histories.
    /// Global slot identities, segment ownership and lifetimes remain unchanged. Existing
    /// active slots require an identical policy; all other slots use ordinary geometry checks.
    pub fn with_materialized_state_slots(
        self,
        state: StateLayout,
        slots: impl IntoIterator<Item = StateSlotId>,
    ) -> Result<Self, ComponentDefinitionError> {
        let mut materialized = std::collections::BTreeSet::new();
        for StateSlotId(slot) in slots {
            let original = self.state.layer(slot);
            let selected = state.layer(slot);
            if !materialized.insert(slot)
                || state
                    .components(slot)
                    .is_none_or(|components| components.is_empty())
                || (!matches!(original, Some(eredu_core::cache::LayerCachePolicy::NoState))
                    && original != selected)
            {
                return Err(ComponentDefinitionError::Invalid {
                    component: self.id,
                    resource: format!("materialized state slot {slot}"),
                });
            }
        }
        self.project_state_layout(state, &materialized)
    }

    fn project_state_layout(
        mut self,
        state: StateLayout,
        materialized: &std::collections::BTreeSet<usize>,
    ) -> Result<Self, ComponentDefinitionError> {
        let compatible = state.len() == self.state.len()
            && state.segments() == self.state.segments()
            && (0..state.len()).all(|slot| {
                if materialized.contains(&slot) {
                    return true;
                }
                let original = self.state.components(slot).expect("validated state slot");
                let selected = state.components(slot).expect("validated state slot");
                self.state.layer(slot).map(|layer| layer.attention())
                    == state.layer(slot).map(|layer| layer.attention())
                    && original.len() == selected.len()
                    && original.iter().zip(selected).all(|(a, b)| {
                        a.role() == b.role()
                            && a.dtype() == b.dtype()
                            && a.residency() == b.residency()
                            && a.presence() == b.presence()
                    })
            });
        if !compatible {
            return Err(ComponentDefinitionError::Invalid {
                component: self.id,
                resource: "selected state identities or lifetimes".into(),
            });
        }
        self.state_partition
            .validate(&state, self.parameters.unit_layout())
            .map_err(|source| ComponentDefinitionError::StatePartition {
                component: self.id.clone(),
                source,
            })?;
        self.state = state;
        Ok(self)
    }
    /// Validates equation resource topology against a family-authored declaration.
    /// Retained rank placement is separate authority and is not reconstructed here.
    pub fn validate_parameter_topology(
        &self,
        expected: &ArchitectureParameterDescription,
    ) -> Result<(), ComponentDefinitionError> {
        let declared = self.parameters();
        let invalid = |resource: String| ComponentDefinitionError::Invalid {
            component: self.id.clone(),
            resource,
        };
        if declared.graph() != expected.graph()
            || declared.unit_layout() != expected.unit_layout()
            || declared.invocations() != expected.invocations()
        {
            return Err(invalid(
                "configured canonical invocation or storage topology".into(),
            ));
        }
        if declared.groups() != expected.groups() {
            let group = declared
                .groups()
                .iter()
                .zip(expected.groups())
                .find_map(|(actual, expected)| (actual != expected).then_some(expected))
                .or_else(|| expected.groups().get(declared.groups().len()))
                .or_else(|| declared.groups().get(expected.groups().len()));
            return Err(invalid(format!(
                "configured canonical parameter group {}",
                group.map_or("ordering", |group| group.group().logical_name()),
            )));
        }
        Ok(())
    }

    /// Checks the equation implementation's semantic projection against this declaration.
    ///
    /// Families supply their unit, transport and state-ownership producers. The
    /// observation predicate admits runtime-owned seams or checks exact boundaries
    /// for instrumented equations without replacing retained observation names.
    pub fn validate_execution_contract(
        &self,
        primary: &str,
        prediction: &[String],
        units: &[ComponentUnit],
        transport: &[ArchitectureGroupTransport],
        state_partition: &ArchitectureStatePartitionPlan,
        observations: impl Fn(
            ExecutionUnitAddress,
            &ComponentUnit,
            &crate::UnitObservationBoundaries,
        ) -> bool,
    ) -> Result<(), ComponentDefinitionError> {
        let invalid = |resource: String| ComponentDefinitionError::Invalid {
            component: self.id.clone(),
            resource,
        };
        if self.primary != primary || self.prediction != prediction {
            return Err(invalid("equation primary or prediction groups".into()));
        }
        if &self.state_partition != state_partition {
            return Err(invalid("equation state ownership".into()));
        }
        if self.transport.len() != transport.len() {
            return Err(invalid("equation transport group count".into()));
        }
        for (group, expected) in transport.iter().enumerate() {
            if &self.transport[group] != expected {
                return Err(invalid(format!("equation group {group} transport")));
            }
        }
        if self.units.len() != units.len() {
            return Err(invalid("equation invocation count".into()));
        }
        for (ordinal, expected) in units.iter().enumerate() {
            let address = self
                .parameters
                .unit_layout()
                .address(ordinal)
                .expect("validated invocation layout");
            if &self.units[ordinal] != expected
                || !observations(address, expected, &self.unit_observations[ordinal])
            {
                return Err(invalid(format!(
                    "equation unit {}:{} path, state references or observation ownership",
                    address.group(),
                    address.index()
                )));
            }
        }
        Ok(())
    }

    /// Projects physical parameters while retaining logical aliases, state and observations.
    /// An identical catalog reuses the retained definition. Changed catalogs undergo
    /// the same ownership and companion checks as [`Self::with_parameters`].
    /// Families that change logical aliases supply them through that consuming method.
    pub fn project_parameters(
        self: &std::sync::Arc<Self>,
        parameters: ArchitectureParameterDescription,
    ) -> Result<std::sync::Arc<Self>, ComponentDefinitionError> {
        self.project_parameter_ownership(parameters, ParameterOwnershipProjection::Exact)
    }

    /// Selects owned physical resources from this retained canonical declaration.
    /// Invocation/storage identities and every selected primary owner remain exact;
    /// omitted primary resources cannot be reintroduced through the resulting subset.
    /// Explicit logical uses must still resolve, so active aliases cannot dangle.
    pub fn project_owned_parameters(
        self: &std::sync::Arc<Self>,
        parameters: ArchitectureParameterDescription,
    ) -> Result<std::sync::Arc<Self>, ComponentDefinitionError> {
        self.project_parameter_ownership(parameters, ParameterOwnershipProjection::Subset)
    }

    fn project_parameter_ownership(
        self: &std::sync::Arc<Self>,
        parameters: ArchitectureParameterDescription,
        projection: ParameterOwnershipProjection,
    ) -> Result<std::sync::Arc<Self>, ComponentDefinitionError> {
        if self.parameters == parameters {
            return Ok(self.clone());
        }
        let mut selected = self.as_ref().clone();
        let uses = std::mem::take(&mut selected.parameter_uses);
        selected
            .with_parameter_ownership(parameters, uses, projection)
            .map(std::sync::Arc::new)
    }

    /// Projects selected physical parameter geometry while retaining semantic identities.
    /// Families supply exact replacement uses, including changed encoding companions.
    pub fn with_parameters(
        self,
        parameters: ArchitectureParameterDescription,
        uses: impl IntoIterator<Item = (String, ComponentParameterUse)>,
    ) -> Result<Self, ComponentDefinitionError> {
        self.with_parameter_ownership(parameters, uses, ParameterOwnershipProjection::Exact)
    }

    fn with_parameter_ownership(
        mut self,
        parameters: ArchitectureParameterDescription,
        uses: impl IntoIterator<Item = (String, ComponentParameterUse)>,
        projection: ParameterOwnershipProjection,
    ) -> Result<Self, ComponentDefinitionError> {
        let invalid = |resource: &str| ComponentDefinitionError::Invalid {
            component: self.id.clone(),
            resource: resource.into(),
        };
        if parameters.graph() != self.parameters.graph()
            || parameters.unit_layout() != self.parameters.unit_layout()
            || parameters.invocations() != self.parameters.invocations()
        {
            return Err(invalid("selected parameter invocation or storage identity"));
        }
        fn primaries(
            description: &ArchitectureParameterDescription,
        ) -> std::collections::BTreeMap<&str, &crate::ParameterGroupOwner> {
            description
                .groups()
                .iter()
                .flat_map(|group| {
                    group
                        .members()
                        .iter()
                        .filter(|member| member.linear_companion().is_none())
                        .map(move |member| (member.target(), group.owner()))
                })
                .collect()
        }
        let original = primaries(&self.parameters);
        let selected = primaries(&parameters);
        if (projection == ParameterOwnershipProjection::Exact && original.len() != selected.len())
            || selected
                .iter()
                .any(|(target, owner)| original.get(target) != Some(owner))
        {
            return Err(invalid("selected primary parameter ownership"));
        }
        for group in parameters.groups() {
            for member in group.members() {
                match (member.linear_companion(), member.linear_companion_of()) {
                    (Some(_), Some(primary))
                        if selected.get(primary).copied() == Some(group.owner()) => {}
                    (None, None) => {}
                    _ => {
                        return Err(invalid(&format!(
                            "selected companion {} has no matching primary owner",
                            member.target()
                        )))
                    }
                }
            }
        }
        self.parameters = parameters;
        self.with_parameter_uses(uses)
    }

    /// Retains architecture-authored logical references after validating their consumers.
    pub fn with_parameter_uses(
        mut self,
        uses: impl IntoIterator<Item = (String, ComponentParameterUse)>,
    ) -> Result<Self, ComponentDefinitionError> {
        let mut declared = std::collections::BTreeMap::new();
        let mut owners = None;
        for (logical, usage) in uses {
            let address = usage.invocation;
            let invalid = || ComponentDefinitionError::Invalid {
                component: self.id.clone(),
                resource: format!("parameter use {logical:?} -> {:?}", usage.target),
            };
            self.unit(address.group(), address.index())?;
            let group = self
                .parameters
                .unit_layout()
                .group_id(address.group())
                .expect("validated group");
            let owned = owners
                .get_or_insert_with(|| {
                    self.parameters
                        .groups()
                        .iter()
                        .flat_map(|owned| {
                            owned.members().iter().map(move |member| (member.target(), owned))
                        })
                        .collect::<std::collections::BTreeMap<_, _>>()
                })
                .get(usage.target.as_str())
                .ok_or_else(&invalid)?;
            let admitted = match owned.owner() {
                crate::ParameterGroupOwner::ExecutionUnit {
                    group: owner,
                    consumers,
                    ..
                } => owner == group && consumers.contains(&address.index()),
                crate::ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => consumers
                    .iter()
                    .any(|(owner, index)| owner == group && *index == address.index()),
                _ => false,
            };
            if logical.trim().is_empty() || !admitted || declared.contains_key(&logical) {
                return Err(invalid());
            }
            declared.insert(logical, usage);
        }
        self.parameter_uses = declared;
        Ok(self)
    }
    /// Exact logical references; these do not introduce physical parameter owners.
    pub fn parameter_uses(&self) -> &std::collections::BTreeMap<String, ComponentParameterUse> {
        &self.parameter_uses
    }
    /// Invocation owning a declared seam or an internal observation below its unit path.
    pub fn observation_invocation(&self, path: &str) -> Option<ExecutionUnitAddress> {
        let original = path.strip_suffix(".effective").unwrap_or(path);
        let exact = self.unit_observations.iter().position(|boundaries| {
            boundaries.input_path() == Some(path)
                || boundaries.output_path() == Some(path)
                || boundaries.input_path() == Some(original)
                || boundaries.output_path() == Some(original)
        });
        let ordinal = exact.or_else(|| {
            self.units
                .iter()
                .enumerate()
                .filter(|(_, unit)| {
                    original
                        .strip_prefix(&unit.path)
                        .is_some_and(|suffix| suffix.starts_with('.'))
                })
                .max_by_key(|(_, unit)| unit.path.len())
                .map(|(ordinal, _)| ordinal)
        })?;
        self.parameters.unit_layout().address(ordinal)
    }
    /// Canonical number of logical invocations in one validated group.
    pub fn unit_count(&self, group: usize) -> Result<usize, ComponentDefinitionError> {
        self.parameters
            .unit_layout()
            .group_range(group)
            .map(|range| range.len())
            .ok_or_else(|| ComponentDefinitionError::Address {
                component: self.id.clone(),
                group,
                index: 0,
            })
    }
    /// Transport declaration for one validated semantic group.
    pub fn group_transport(
        &self,
        group: usize,
    ) -> Result<&ArchitectureGroupTransport, ComponentDefinitionError> {
        self.transport
            .get(group)
            .ok_or_else(|| ComponentDefinitionError::Address {
                component: self.id.clone(),
                group,
                index: 0,
            })
    }
    /// Validates a group invocation against the retained dependency declaration.
    pub fn validate_group_dependencies(
        &self,
        group: usize,
        count: usize,
    ) -> Result<(), ComponentDefinitionError> {
        if self
            .parameters
            .graph()
            .dependencies(group)
            .is_none_or(|declared| declared.len() != count)
        {
            return Err(ComponentDefinitionError::Invalid {
                component: self.id.clone(),
                resource: format!("dependencies for group {group}"),
            });
        }
        Ok(())
    }
    /// Stable identity of the primary semantic group.
    pub fn primary_group(&self) -> &str {
        &self.primary
    }
    /// Ordered prediction groups, independent of declaration ordinals.
    pub fn prediction_groups(&self) -> &[String] {
        &self.prediction
    }
    /// State ownership projected from the same retained declaration.
    pub fn state_partition_plan(&self) -> &ArchitectureStatePartitionPlan {
        &self.state_partition
    }
    /// Projects exact per-operation seams and ownership before execution selection.
    /// Boundary names are globally unique within this component.
    pub fn map_unit_observation_boundaries(
        mut self,
        mut project: impl FnMut(
            ExecutionUnitAddress,
            &crate::UnitObservationBoundaries,
        ) -> crate::UnitObservationBoundaries,
    ) -> Result<Self, ComponentDefinitionError> {
        let projected = self
            .unit_observations
            .iter()
            .enumerate()
            .map(|(ordinal, boundaries)| {
                project(
                    self.parameters
                        .unit_layout()
                        .address(ordinal)
                        .expect("declared invocation"),
                    boundaries,
                )
            })
            .collect::<Vec<_>>();
        let mut paths = std::collections::BTreeSet::new();
        for boundaries in &projected {
            for path in [boundaries.input_path(), boundaries.output_path()]
                .into_iter()
                .flatten()
            {
                if path.trim().is_empty() || !paths.insert(path) {
                    return Err(ComponentDefinitionError::Invalid {
                        component: self.id.clone(),
                        resource: format!("observation boundary {path:?}"),
                    });
                }
            }
        }
        self.unit_observations = projected;
        Ok(self)
    }

    /// Borrows the retained logical boundaries of one exact invocation.
    pub fn unit_observation_boundaries(
        &self,
        group: usize,
        index: usize,
    ) -> Result<&crate::UnitObservationBoundaries, ComponentDefinitionError> {
        let ordinal = self
            .parameters
            .unit_layout()
            .ordinal(group, index)
            .ok_or_else(|| ComponentDefinitionError::Address {
                component: self.id.clone(),
                group,
                index,
            })?;
        Ok(&self.unit_observations[ordinal])
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
                    group, consumers, ..
                } => group == group_id && consumers.contains(&index),
                crate::ParameterGroupOwner::StaticUnitConsumers { consumers, .. } => consumers
                    .iter()
                    .any(|(group, unit)| group == group_id && *unit == index),
                _ => false,
            })
            .map(|owned| ParameterStorageId::from_owner(owned.owner()))
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

/// Typed resource construction and request roots for independently leased modules.
/// Implementing this trait alone declares no parallel or partition capability.
pub trait ComponentKernel<B: NeuralBackend> {
    /// Family-authored checkpoint rewrites for pinned canonical resources.
    fn static_parameter_recipes(
        &self,
        _source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        Ok(std::collections::BTreeMap::new())
    }
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
    /// Meaning of this invocation's complete output. Context or feature-only
    /// operations return `None` and bypass the model-score observation seam.
    fn output_score_kind(
        _forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        Some(eredu_core::refinement::PredictionScoreKind::CausalConditional)
    }
    /// Borrows the exact target activation retained for an external prediction operation.
    fn prediction_target_capture(_forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        None
    }
    /// Exact capture geometry for a partition that does not publish target output.
    fn prediction_target_placeholder_shape(
        &self,
        _forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(None)
    }
    /// Builds one canonical parameter resource after shared storage validation.
    /// This address identifies its physical owner; equations receive logical consumers.
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error>;

    /// Operation-specific group activity.
    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool;

    /// Retains operation inputs and candidates until native completion is established.
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a>;
}

/// Typed equations over one concrete borrowed or owned state profile.
/// Resource construction and request roots remain independent of cache storage ownership.
pub trait ComponentEquations<B: NeuralBackend, S: RuntimeState<B>>: ComponentKernel<B> {
    /// Explicit coverage of ordinary internal equation hooks.
    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Optional semantic path for the selected group input.
    fn group_input_observation_path(&self, _group: usize) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    /// Optional semantic path for the completed group output.
    fn group_output_observation_path(&self, _group: usize) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    /// Runs ordinary ingress with optional internal observation and intervention.
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin(input, state, context)
    }

    /// Runs one validated unit with optional internal observation and intervention.
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.execute(address, unit, hidden, state, forward, context)
    }

    /// Runs ordinary readout with optional internal observation and intervention.
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish(hidden, state, forward, context)
    }

    /// Whether causal-text readout projects only the final hidden position.
    fn projects_final_text_position() -> bool {
        false
    }

    /// Readout for an unobserved text consumer that needs only the final position.
    /// Other operations and observed readout retain the ordinary result geometry.
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish(hidden, state, forward, context)
    }

    /// Prepares a typed operation.
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Selects or assembles group input after shared dependency-count validation.
    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Closes one validated group and retains its dependency-facing values.
    fn complete_group(
        &mut self,
        _group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden.clone())
    }
    /// Runs the family equation against an already admitted resource.
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Converts the final activation to the typed operation's ordinary tensor result.
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Explicit parallel equations; availability is enforced by a trait bound.
pub trait ParallelComponentKernel<B: NeuralBackend, S: RuntimeState<B>>:
    ComponentEquations<B, S>
{
    /// Explicit coverage of parallel internal equation hooks.
    fn parallel_observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Runs parallel ingress with optional internal observation and intervention.
    fn begin_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_parallel(input, state, parallel, context)
    }

    /// Runs one validated parallel unit with optional internal instrumentation.
    fn execute_parallel_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.execute_parallel(address, unit, hidden, state, forward, parallel, context)
    }

    /// Runs parallel readout with optional internal observation and intervention.
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish_parallel(hidden, state, forward, parallel, context)
    }

    /// Parallel ingress and communicator validation.
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;
    /// Enters a validated group at its architecture-owned collective wave.
    fn begin_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.begin_group(group, initial, dependencies, state, forward, context)
    }
    /// Closes the same group under an admitted parallel context.
    fn complete_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.complete_group(group, hidden, state, forward, context)
    }
    /// Parallel unit equation, including architecture-specific collectives.
    fn execute_parallel(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
    /// Parallel readout.
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;
}

/// Validated invocation, canonical storage and independent mutable-state ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentSelection {
    component: String,
    invocations: crate::ExecutionUnitAddressMap,
    units: Vec<ExecutionUnitAddress>,
    storage: std::collections::BTreeSet<ParameterStorageId>,
    state: Option<PartitionState>,
    owns_output: bool,
}

impl ComponentSelection {
    /// Selected logical invocations and independently leased physical owners.
    pub fn invocations(&self) -> &crate::ExecutionUnitAddressMap {
        &self.invocations
    }
    /// Whether canonical parameter storage is physically selected on this partition.
    pub fn owns_storage(&self, storage: &ParameterStorageId) -> bool {
        self.storage.contains(storage)
    }
    /// Selected mutable state, including explicit global-ordinal gaps.
    pub fn state(&self) -> Option<&PartitionState> {
        self.state.as_ref()
    }
    /// Resolves a retained owned invocation to its architecture-global address.
    pub fn address(
        &self,
        group: usize,
        index: usize,
    ) -> Result<ExecutionUnitAddress, ComponentDefinitionError> {
        self.units
            .binary_search_by_key(&(group, index), |address| (address.group(), address.index()))
            .map(|ordinal| self.units[ordinal])
            .map_err(|_| ComponentDefinitionError::Address {
                component: self.component.clone(),
                group,
                index,
            })
    }
}

impl ComponentDefinition {
    /// Validates canonical parameter, unit and mutable-state ownership before materialization.
    pub fn select_partition<G, A>(
        &self,
        partition: &crate::ArchitecturePartition<G, A>,
    ) -> Result<ComponentSelection, ComponentDefinitionError> {
        let definition = self;
        let invalid = |resource: &str| ComponentDefinitionError::Invalid {
            component: definition.id.clone(),
            resource: resource.into(),
        };
        if partition.graph() != definition.parameters.graph()
            || partition.unit_layout() != definition.parameters.unit_layout()
            || partition.parameter_bindings() != definition.parameters.select_owned(partition)
            || !definition
                .parameters
                .invocations()
                .select_invocations(partition.units())
                .is_ok_and(|selected| &selected == partition.invocations())
        {
            return Err(invalid("partition parameter ownership"));
        }
        let expected = definition
            .state_partition
            .resolve_partition(
                &definition.state,
                definition.parameters.unit_layout(),
                |group| {
                    partition
                        .groups()
                        .iter()
                        .find(|owned| owned.group_index() == group)
                        .map(|owned| owned.global_units())
                },
                partition.ownership().owns_output(),
            )
            .map_err(|source| ComponentDefinitionError::StatePartition {
                component: definition.id.clone(),
                source,
            })?;
        if partition.state() != expected.as_ref() {
            return Err(invalid("partition state ownership"));
        }
        // Partition traversal supplies canonical group/unit order for binary lookup.
        let mut units = Vec::with_capacity(partition.invocations().logical_addresses().len());
        for address in partition.units() {
            let unit = definition.unit(address.group(), address.index())?;
            if !unit.state.is_empty()
                && expected.as_ref().is_none_or(|state| {
                    unit.state
                        .clone()
                        .any(|slot| !state.owns_global_layer(slot))
                })
            {
                return Err(invalid("partition unit state references"));
            }
            units.push(address);
        }
        let selected = ComponentSelection {
            component: self.id.clone(),
            invocations: partition.invocations().clone(),
            units,
            storage: partition
                .parameter_bindings()
                .iter()
                .map(|group| ParameterStorageId::from_owner(group.owner()))
                .collect(),
            state: expected,
            owns_output: partition.ownership().owns_output(),
        };
        Ok(selected)
    }
    /// Resolves one canonical global invocation without projecting its identity to a local ordinal.
    pub fn unit_address(
        &self,
        group: usize,
        index: usize,
    ) -> Result<ExecutionUnitAddress, ComponentDefinitionError> {
        let layout = self.parameters.unit_layout();
        layout
            .ordinal(group, index)
            .and_then(|ordinal| layout.address(ordinal))
            .ok_or_else(|| ComponentDefinitionError::Address {
                component: self.id.clone(),
                group,
                index,
            })
    }
}

/// Shared driver-facing contracts over one declaration and typed family kernel.
#[derive(Clone)]
pub struct ComponentAdapter<D, K, R = EmptyParameterModule> {
    definition: D,
    kernel: K,
    resources: R,
    resource_roles: std::collections::BTreeSet<String>,
    selection: Option<ComponentSelection>,
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
        Ok(Self::from_resources(
            definition,
            kernel,
            EmptyParameterModule,
            Default::default(),
        ))
    }
}
impl<D: Borrow<ComponentDefinition>, K, R> ComponentAdapter<D, K, R> {
    pub(crate) fn from_resources(
        definition: D,
        kernel: K,
        resources: R,
        resource_roles: std::collections::BTreeSet<String>,
    ) -> Self {
        Self {
            definition,
            kernel,
            resources,
            resource_roles,
            selection: None,
        }
    }
    pub(crate) fn validate_resource_roles(
        &self,
        groups: &[crate::OwnedParameterGroupSpec],
    ) -> Result<(), ComponentDefinitionError> {
        let expected: std::collections::BTreeSet<_> = groups
            .iter()
            .filter_map(|group| group.owner().static_storage_role())
            .map(str::to_owned)
            .collect();
        if self.resource_roles != expected {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: format!(
                    "pinned roles {:?}, expected {expected:?}",
                    self.resource_roles
                ),
            });
        }
        Ok(())
    }
    /// Consumes the adapter and transfers its retained definition and resource owners.
    pub fn into_parts(self) -> (D, K, R) {
        (self.definition, self.kernel, self.resources)
    }

    /// Retains validated rank-local ownership without changing global invocation identities.
    pub fn select_partition<G, A>(
        mut self,
        partition: &crate::ArchitecturePartition<G, A>,
    ) -> Result<Self, ComponentDefinitionError> {
        let selected = self.definition().select_partition(partition)?;
        self.validate_resource_roles(partition.parameter_bindings())?;
        let invalid = |resource: &str| ComponentDefinitionError::Invalid {
            component: self.definition().id.clone(),
            resource: resource.into(),
        };
        if self
            .selection
            .as_ref()
            .is_some_and(|retained| retained != &selected)
        {
            return Err(invalid("incompatible retained partition"));
        }
        self.selection = Some(selected);
        Ok(self)
    }
    pub(crate) fn validate_partition_group(
        &self,
        group: usize,
    ) -> Result<(), ComponentDefinitionError> {
        if self
            .selection
            .as_ref()
            .is_none_or(|selection| !selection.units.iter().any(|address| address.group() == group))
        {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: format!("partition group {group}"),
            });
        }
        Ok(())
    }
    fn group_has_state_geometry(&self, group: usize) -> bool {
        let definition = self.definition();
        if definition.state_partition.rules().iter().any(|rule| {
            matches!(rule.placement(), crate::ArchitectureStatePlacement::GroupUnits { group: owner } if owner == group)
        }) {
            return true;
        }
        let layout = definition.parameters.unit_layout();
        let Some(mut units) = layout.group_range(group) else {
            return false;
        };
        units.any(|ordinal| !definition.units[ordinal].state.is_empty())
    }
    /// Constructs the ordinary local-stage traversal from declared state consumers.
    pub fn into_partition_runtime<B, S, G, A, P>(
        self,
        policy: P,
        partition: &crate::ArchitecturePartition<G, A>,
    ) -> Result<
        crate::LayerwisePartitionRuntime<Self, B, S, P>,
        crate::LayerwisePartitionConstructionError,
    >
    where
        B: NeuralBackend,
        D: 'static,
        R: ComponentResourceAccess<B, K> + 'static,
        S: RuntimeState<B>,
        K: crate::PartitionComponentKernel<B, S, Boundary = A> + 'static,
        A: crate::ArchitectureBoundary,
        P: crate::LayerwisePolicy<B, K::Unit>,
    {
        let selected = self.select_partition(partition)?;
        let definition = selected.definition();
        let layout = definition.parameters.unit_layout();
        let mut state_groups = std::collections::BTreeMap::new();
        for owned in partition.groups() {
            let group = owned.group_index();
            if !selected.group_has_state_geometry(group) {
                continue;
            }
            let units = owned.global_units();
            let state = if let Some(range) =
                definition
                    .state_partition
                    .group_state_range(layout, group, units.clone())?
            {
                range
            } else {
                let mut ranges = units
                    .map(|index| {
                        definition
                            .unit(group, index)
                            .expect("declared unit")
                            .state
                            .clone()
                    })
                    .filter(|range| !range.is_empty());
                let Some(first) = ranges.next() else {
                    continue;
                };
                ranges.fold(first, |span, range| {
                    span.start.min(range.start)..span.end.max(range.end)
                })
            };
            state_groups.insert(group, state);
        }
        crate::LayerwisePartitionRuntime::new(selected, policy, partition, |group| {
            state_groups.get(&group).cloned()
        })
    }
    pub(crate) fn partition_entry_layout<'a>(
        &self,
        expected: Option<&'a StateLayout>,
    ) -> Option<&'a StateLayout> {
        if self
            .selection
            .as_ref()
            .is_some_and(|selection| selection.state.is_none())
            && expected == Some(&self.definition().state)
        {
            None
        } else {
            expected
        }
    }
    /// A typed partition boundary can resume local state after stateless groups.
    /// Pipeline ingress still validates the first selected group's optional geometry.
    pub(crate) fn validate_partition_resume_state(
        &self,
        actual: Option<&StateLayout>,
        expected: Option<&StateLayout>,
    ) -> Result<(), ComponentDefinitionError> {
        self.validate_partition_state(actual, None, None)?;
        let declared = self
            .selection
            .as_ref()
            .and_then(|selection| selection.state.as_ref())
            .map(PartitionState::layout);
        if expected != declared {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: "partition boundary state layout".into(),
            });
        }
        Ok(())
    }
    pub(crate) fn validate_partition_state(
        &self,
        actual: Option<&StateLayout>,
        expected_entry: Option<Option<&StateLayout>>,
        owns_output: Option<bool>,
    ) -> Result<(), ComponentDefinitionError> {
        let selected =
            self.selection
                .as_ref()
                .ok_or_else(|| ComponentDefinitionError::Invalid {
                    component: self.definition().id.clone(),
                    resource: "missing retained partition".into(),
                })?;
        let declared = selected.state.as_ref().map(PartitionState::layout);
        let entry_declared = selected.units.first().and_then(|address| {
            self.group_has_state_geometry(address.group())
                .then_some(declared)
                .flatten()
        });
        let actual_matches = actual == declared
            || (actual.is_none()
                && declared.is_some_and(|layout| {
                    layout.layers().iter().all(|policy| {
                        matches!(policy, eredu_core::cache::LayerCachePolicy::NoState)
                    })
                }));
        if !actual_matches
            || expected_entry.is_some_and(|expected| expected != entry_declared)
            || owns_output.is_some_and(|owns| owns != selected.owns_output)
        {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: "partition state or output ownership".into(),
            });
        }
        Ok(())
    }
    pub(crate) fn local_state_slots(&self, group: usize, index: usize) -> Range<usize> {
        let slots = self
            .definition()
            .unit(group, index)
            .expect("validated traversal address")
            .state
            .clone();
        if slots.is_empty() {
            return 0..0;
        }
        match self
            .selection
            .as_ref()
            .and_then(|selection| selection.state.as_ref())
        {
            Some(state) => {
                (slots.start - state.global_layer_offset())
                    ..(slots.end - state.global_layer_offset())
            }
            None => slots,
        }
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
    pub(crate) fn address(
        &self,
        group: usize,
        index: usize,
    ) -> Result<ExecutionUnitAddress, ComponentDefinitionError> {
        match &self.selection {
            Some(selection) => selection.address(group, index),
            None => self.definition().unit_address(group, index),
        }
    }
}
impl<B, D, K, R> ArchitectureParameters<B> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
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
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        self.kernel.static_parameter_recipes(source)
    }
    fn visit_static_parameters<V: StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.resources.visit(&self.kernel, visitor)
    }
    fn visit_static_parameters_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.resources.visit_mut(&mut self.kernel, visitor)
    }
}
impl<B, D, K, S, R> LayeredArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: ComponentEquations<B, S> + 'static,
{
    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        K::output_score_kind(forward)
    }
    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        self.kernel.observation_hooks()
    }

    fn unit_observation_boundaries(
        &self,
        group: usize,
        index: usize,
    ) -> Result<crate::UnitObservationBoundaries, K::Error> {
        self.definition()
            .unit_observation_boundaries(group, index)
            .cloned()
            .map_err(Into::into)
    }

    fn group_input_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        self.definition().unit_count(group)?;
        self.kernel.group_input_observation_path(group)
    }

    fn group_output_observation_path(&self, group: usize) -> Result<Option<String>, Self::Error> {
        self.definition().unit_count(group)?;
        self.kernel.group_output_observation_path(group)
    }

    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel.begin_observed(input, state, context, observer)
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel.execute_observed(
            self.address(group, index)?,
            unit,
            hidden,
            state,
            forward,
            context,
            observer,
        )
    }

    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel
            .finish_observed(hidden, state, forward, context, observer)
    }

    type Input<'a>
        = K::Input<'a>
    where
        Self: 'a;
    type StaticModules = R::Modules;
    type Unit = K::Unit;
    type ForwardContext = K::ForwardContext;
    type RetainedContextValues<'a>
        = K::RetainedContextValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = K::Error;
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport {
        self.definition()
            .group_transport(group)
            .expect("validated traversal group")
            .clone()
    }
    fn primary_execution_group(&self) -> &str {
        self.definition().primary_group()
    }
    fn prediction_execution_groups(&self) -> Vec<String> {
        self.definition().prediction_groups().to_vec()
    }
    fn state_partition_plan(&self, _: &StateLayout) -> ArchitectureStatePartitionPlan {
        self.definition().state_partition_plan().clone()
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, K::Error> {
        Ok(self.definition().parameters.graph().clone())
    }
    fn execution_unit_layout(
        &self,
    ) -> Result<crate::ExecutionUnitLayout, crate::ArchitectureExecutionLayoutError<K::Error>> {
        Ok(self.definition().parameters.unit_layout().clone())
    }
    fn execution_unit_storage_map(
        &self,
    ) -> Result<crate::ExecutionUnitAddressMap, crate::ArchitectureExecutionLayoutError<K::Error>>
    {
        Ok(self.definition().parameters.invocations().clone())
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, K::Error> {
        self.definition().unit_count(group).map_err(Into::into)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, K::Error> {
        Ok(self.definition().unit(group, index)?.path.clone())
    }
    fn static_modules(&self) -> &R::Modules {
        self.resources.modules(&self.kernel)
    }
    fn static_modules_mut(&mut self) -> &mut R::Modules {
        self.resources.modules_mut(&mut self.kernel)
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<K::Unit, K::Error> {
        let address = self.address(group, index)?;
        let ordinal = self
            .definition()
            .parameters
            .unit_layout()
            .ordinal(group, index)
            .expect("validated invocation");
        let storage = self
            .definition()
            .parameters
            .invocations()
            .resolve(ordinal, address)
            .expect("validated declaration storage")
            .storage_location();
        self.kernel.build_unit(storage.address(), context)
    }
    fn build_storage_unit(
        &self,
        storage: crate::ExecutionUnitStorageLocation,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<K::Unit, K::Error> {
        let address = storage.address();
        let invocations = self.selection.as_ref().map_or_else(
            || self.definition().parameters.invocations(),
            |selection| selection.invocations(),
        );
        if invocations
            .storage_layout()
            .ordinal(address.group(), address.index())
            .is_none()
        {
            return Err(ComponentDefinitionError::Invalid {
                component: self.definition().id.clone(),
                resource: format!("physical owner {address:?}"),
            }
            .into());
        }
        self.kernel.build_unit(address, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, K::ForwardContext>, K::Error> {
        self.kernel.begin(input, state, context)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.definition()
            .validate_group_dependencies(group, dependencies.len())?;
        self.kernel
            .begin_group(group, initial, dependencies, state, forward, context)
    }
    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.definition().unit_count(group)?;
        self.kernel
            .complete_group(group, hidden, state, forward, context)
    }
    fn prediction_target_capture(forward: &K::ForwardContext) -> Option<&B::Tensor> {
        K::prediction_target_capture(forward)
    }
    fn prediction_target_placeholder_shape(
        &self,
        forward: &K::ForwardContext,
    ) -> Result<Option<Vec<i32>>, K::Error> {
        self.kernel.prediction_target_placeholder_shape(forward)
    }
    fn should_execute_group(&self, group: usize, forward: &K::ForwardContext) -> bool {
        self.kernel.executes_group(group, forward)
    }
    fn retained_state_ordinals(&self, group: usize, index: usize, _: usize) -> Range<usize> {
        self.local_state_slots(group, index)
    }
    fn state_ordinal(&self, group: usize, index: usize, _: usize) -> usize {
        self.local_state_slots(group, index).start
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut K::Unit,
        hidden: &B::Tensor,
        state: &mut S,
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
        state: &mut S,
        forward: &K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel.finish(hidden, state, forward, context)
    }
    fn projects_final_text_position() -> bool {
        K::projects_final_text_position()
    }
    fn finish_text_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &K::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel.finish_text(hidden, state, forward, context)
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
impl<B, D, K, S, R> ParallelLayeredArchitecture<B, S> for ComponentAdapter<D, K, R>
where
    B: NeuralBackend,
    D: Borrow<ComponentDefinition> + 'static,
    R: ComponentResourceAccess<B, K> + 'static,
    S: RuntimeState<B>,
    K: ParallelComponentKernel<B, S> + 'static,
{
    fn parallel_observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        self.kernel.parallel_observation_hooks()
    }

    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel
            .begin_parallel_observed(input, state, parallel, context, observer)
    }

    fn forward_unit_parallel_observed<O>(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel.execute_parallel_observed(
            self.address(group_index, index)?,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            observer,
        )
    }

    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.kernel
            .finish_parallel_observed(hidden, state, forward, parallel, context, observer)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
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
        state: &mut S,
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
    fn begin_execution_group_parallel(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut K::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.definition()
            .validate_group_dependencies(group, dependencies.len())?;
        self.kernel.begin_group_parallel(
            group,
            initial,
            dependencies,
            state,
            forward,
            parallel,
            context,
        )
    }
    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut K::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.definition().unit_count(group)?;
        self.kernel
            .complete_group_parallel(group, hidden, state, forward, parallel, context)
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &K::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, K::Error> {
        self.kernel
            .finish_parallel(hidden, state, forward, parallel, context)
    }
}
