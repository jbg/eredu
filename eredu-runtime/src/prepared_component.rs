//! Exact cold component bindings and consuming placement selection.

use crate::{
    place_weight_bindings, prepare_resource_bindings, ArchitectureParameterDescription,
    BindingPlacementError, ExecutionUnitAddress, ExecutionUnitLayout, ExecutionUnitLayoutError,
    LocalModelLayout, ParameterGroupOwner, ParameterMemberSpec, ParameterResourceRecipes,
    ParameterStorageId, RecipeBindingError, WeightBinding, WeightBindingPlan,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeAlias, RecipeDtype, RecipeError},
    store::{CheckpointSource, RestrictedCheckpointSource, SharedCheckpointSource, StoreError},
};
use std::{collections::BTreeMap, sync::Arc};

/// Failure preparing exact component resources without allocating native values.
#[derive(Debug, thiserror::Error)]
pub enum ComponentPreparationError {
    /// Selected unit identities are not a canonical subset.
    #[error(transparent)]
    Units(#[from] ExecutionUnitLayoutError),
    /// Selected invocation references are incompatible with retained storage.
    #[error(transparent)]
    Invocations(#[from] crate::ExecutionUnitAddressMapError),
    /// Source authority cannot be restricted to the declared physical inputs.
    #[error(transparent)]
    Source(#[from] StoreError),
    /// A recipe cannot be inferred against its retained source.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// A recipe cannot be represented as a binding.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Logical consumers of one materialization select different physical slices.
    #[error("aliased parameter {alias:?} has different placement from owner {owner:?}")]
    AliasPlacement {
        /// Logical consumer target.
        alias: String,
        /// Canonical materialization target.
        owner: String,
    },
    /// The retained declaration cannot project the requested parallel topology.
    #[error("component {component:?} cannot derive parallel placement: {reason}")]
    ParallelLayout {
        /// Component identity.
        component: String,
        /// Exact neutral planner failure.
        reason: String,
    },
    /// Component tensor selection cannot own pipeline, expert or data dimensions.
    #[error("component {component:?} tensor selection requires a tensor-only topology, got {topology:?}")]
    TensorOnlyTopology {
        /// Retained component identity.
        component: String,
        /// Rejected exact rank topology.
        topology: eredu_core::ParallelRankTopology,
    },
    /// A selected placement cannot be applied.
    #[error(transparent)]
    Placement(#[from] BindingPlacementError),
    /// A canonical target has geometry different from its declaration.
    #[error("component {component:?} parameter {target:?} has inconsistent geometry")]
    Geometry {
        /// Component identity.
        component: String,
        /// Canonical parameter target.
        target: String,
    },
    /// A family-selected output dtype cannot be reached through its allowed casts.
    #[error(
        "component {component:?} parameter {target:?} produces {actual:?}, expected {expected:?}"
    )]
    Dtype {
        /// Component identity.
        component: String,
        /// Canonical parameter target.
        target: String,
        /// Inferred recipe output before conversion.
        actual: RecipeDtype,
        /// Family-selected execution representation.
        expected: RecipeDtype,
    },
    /// The caller requests unit-only parts for a component with pinned parameters.
    #[error("component {0:?} has pinned resources")]
    PinnedResources(String),
    /// A retained placement disagrees with the requested selection.
    #[error("component {0:?} already retains a different placement")]
    SelectionMismatch(String),
    /// Native storage preferences describe a different invocation graph.
    #[error("component {0:?} storage preferences use a different architecture")]
    ArchitectureMismatch(String),
    /// A requested projection is not a subset of the retained definition.
    #[error("component {component:?} projection changes parameter {target:?}")]
    InvalidProjection {
        /// Component identity.
        component: String,
        /// Missing or altered canonical parameter.
        target: String,
    },
}

impl From<ComponentPreparationError> for eredu_nn::Error {
    fn from(error: ComponentPreparationError) -> Self {
        Self::backend_source(error)
    }
}

/// Family-selected output dtype and explicitly admitted source conversions.
/// The target representation is retained without adding a redundant cast.
#[derive(Debug, Clone)]
pub struct ComponentRecipeConversion {
    /// Required output of initial recipe normalization.
    pub dtype: RecipeDtype,
    /// Other admitted recipe output dtypes that may be cast to the target.
    pub from: &'static [RecipeDtype],
}
impl ComponentRecipeConversion {
    fn apply(
        self,
        recipe: DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        component: &str,
        target: &str,
    ) -> Result<DerivedWeightRecipe, ComponentPreparationError> {
        let actual = recipe.infer(source)?.dtype;
        if actual == self.dtype {
            return Ok(recipe);
        }
        if !self.from.contains(&actual) {
            return Err(ComponentPreparationError::Dtype {
                component: component.into(),
                target: target.into(),
                actual,
                expected: self.dtype,
            });
        }
        Ok(DerivedWeightRecipe::Cast {
            input: Box::new(recipe),
            dtype: self.dtype,
        })
    }
}

/// Family-resolved physical recipe or logical alias within a leased resource.
pub enum ComponentRecipe {
    /// One canonical materialization with an optional family-selected conversion.
    Physical {
        /// Admitted source recipe before execution conversion.
        recipe: DerivedWeightRecipe,
        /// Exact output requirement and allowed casts, when selected by the family.
        conversion: Option<ComponentRecipeConversion>,
    },
    /// A logical parameter shares the named canonical target's materialization.
    Alias(String),
}
impl From<DerivedWeightRecipe> for ComponentRecipe {
    fn from(recipe: DerivedWeightRecipe) -> Self {
        Self::Physical {
            recipe,
            conversion: None,
        }
    }
}

/// Typed declaration retained by exact resource preparation. A resource-only
/// declaration supplies parameter ownership; a full component definition also
/// retains execution, state and boundary semantics through the same algorithm.
pub trait ComponentPreparationDeclaration {
    /// Authoritative canonical parameter ownership and global invocation layout.
    fn parameters(&self) -> &ArchitectureParameterDescription;
}
impl ComponentPreparationDeclaration for ArchitectureParameterDescription {
    fn parameters(&self) -> &ArchitectureParameterDescription {
        self
    }
}
impl ComponentPreparationDeclaration for crate::ComponentDefinition {
    fn parameters(&self) -> &ArchitectureParameterDescription {
        self.parameters()
    }
}

impl<D: ComponentPreparationDeclaration> ComponentPreparationDeclaration for Arc<D> {
    fn parameters(&self) -> &ArchitectureParameterDescription {
        self.as_ref().parameters()
    }
}

/// Consuming exact bindings with restricted source authority and retained placement.
///
/// Family code supplies admitted recipes, including precision policy and special
/// transforms. This container validates global geometry, groups physical ownership,
/// restricts source access, and applies rank-local placement at most once.
pub struct PreparedComponent<D: ComponentPreparationDeclaration> {
    id: String,
    declaration: D,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    selected_units: Option<ExecutionUnitLayout>,
    invocations: crate::ExecutionUnitAddressMap,
    pinned: BTreeMap<ParameterStorageId, Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedComponent<crate::ComponentDefinition> {
    /// Retains selected state geometry without changing resource order, authority or placement.
    pub fn with_state_layout(
        mut self,
        state: crate::StateLayout,
    ) -> Result<Self, crate::ComponentDefinitionError> {
        self.declaration = self.declaration.with_state_layout(state)?;
        Ok(self)
    }
}

impl<D: ComponentPreparationDeclaration> PreparedComponent<D> {
    /// Prepares canonical physical recipes from an admitted catalog. Only declared
    /// targets are resolved through a borrowed lookup; each used recipe is cloned once.
    /// Conversion policy and missing-target errors belong to the caller. Source
    /// restriction, ownership and placement use ordinary preparation.
    pub fn prepare_physical_recipes<'a, E>(
        id: impl Into<String>,
        declaration: D,
        source: SharedCheckpointSource,
        mut recipes: impl FnMut(&str) -> Option<&'a DerivedWeightRecipe>,
        conversion: Option<ComponentRecipeConversion>,
        mut missing: impl FnMut(&str) -> E,
    ) -> Result<Self, E>
    where
        E: From<ComponentPreparationError>,
    {
        Self::prepare(id, declaration, source, |member| {
            let recipe = recipes(member.target())
                .cloned()
                .ok_or_else(|| missing(member.target()))?;
            Ok::<_, E>(ComponentRecipe::Physical {
                recipe,
                conversion: conversion.clone(),
            })
        })
    }

    /// Resolves canonical targets using family-owned admitted recipe policy.
    /// No checkpoint payload is read.
    pub fn prepare<E, R>(
        id: impl Into<String>,
        declaration: D,
        source: SharedCheckpointSource,
        mut resolve: impl FnMut(&ParameterMemberSpec) -> Result<R, E>,
    ) -> Result<Self, E>
    where
        E: From<ComponentPreparationError>,
        R: Into<ComponentRecipe>,
    {
        let id = id.into();
        let description = declaration.parameters();
        let mut resources = BTreeMap::<ParameterStorageId, ParameterResourceRecipes>::new();
        for owned in description.groups() {
            let resource = resources
                .entry(ParameterStorageId::from_owner(owned.owner()))
                .or_insert_with(|| ParameterResourceRecipes {
                    outputs: BTreeMap::new(),
                    aliases: Vec::new(),
                });
            for member in owned.members() {
                match resolve(member)?.into() {
                    ComponentRecipe::Physical { recipe, conversion } => {
                        let recipe = match conversion {
                            Some(conversion) => {
                                conversion.apply(recipe, source.as_ref(), &id, member.target())?
                            }
                            None => recipe,
                        };
                        resource.outputs.insert(member.target().into(), recipe);
                    }
                    ComponentRecipe::Alias(target) => {
                        resource
                            .aliases
                            .push(RecipeAlias::new(member.target(), target));
                    }
                }
            }
        }
        let (source, resources) = prepare_resource_bindings(source, &id, resources)
            .map_err(ComponentPreparationError::from)?;
        // Targets are globally unique; aliases resolve within each physical resource.
        let mut resolved = BTreeMap::new();
        for bindings in resources.values() {
            let plan = WeightBindingPlan::new(bindings)
                .map_err(RecipeBindingError::from)
                .map_err(ComponentPreparationError::from)?;
            resolved.extend(
                bindings
                    .iter()
                    .filter(|binding| !binding.is_alias())
                    .map(|binding| (binding.name(), binding)),
            );
            resolved.extend(
                plan.alias_indices()
                    .map(|(alias, owner)| (bindings[alias].name(), &bindings[owner])),
            );
        }
        for group in description.groups() {
            for member in group.members() {
                let binding = resolved[member.target()];
                if binding
                    .source_recipe()
                    .infer(source.as_ref())
                    .map_err(ComponentPreparationError::from)?
                    .shape
                    != member.global_shape()
                {
                    return Err(ComponentPreparationError::Geometry {
                        component: id,
                        target: member.target().into(),
                    }
                    .into());
                }
            }
        }
        let mut units = vec![Vec::new(); description.invocations().storage_layout().len()];
        let mut pinned = BTreeMap::new();
        for (owner, bindings) in resources {
            let bindings = bindings
                .into_iter()
                .map(|binding| {
                    let target = binding.name().to_owned();
                    binding
                        .with_logical_target(target)
                        .map_err(RecipeBindingError::from)
                        .map_err(ComponentPreparationError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let ParameterStorageId::ExecutionUnit { group, global_unit } = &owner {
                let group = description
                    .graph()
                    .group_index(group.as_str())
                    .expect("validated parameter owner");
                let ordinal = description
                    .invocations()
                    .storage_layout()
                    .ordinal(group, *global_unit)
                    .expect("validated execution unit");
                units[ordinal] = bindings;
            } else {
                pinned.insert(owner, bindings);
            }
        }
        let invocations = description.invocations().clone();
        Ok(Self {
            id,
            declaration,
            source,
            units,
            selected_units: None,
            invocations,
            pinned,
            local_layout: None,
        })
    }

    /// Validated canonical ownership used by this exact preparation.
    pub fn description(&self) -> &ArchitectureParameterDescription {
        self.declaration.parameters()
    }
    /// Exact typed declaration retained before source restriction or placement.
    pub fn declaration(&self) -> &D {
        &self.declaration
    }
    /// Exact retained source authority.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Independently leased physical owners in canonical storage order.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.units
    }
    /// Canonical physical addresses paired with independently leased bindings.
    /// Dense storage ordinals never replace group or global unit identities.
    pub fn addressed_unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (ExecutionUnitAddress, &[WeightBinding])> {
        self.units.iter().enumerate().map(|(ordinal, bindings)| {
            (
                self.invocations
                    .storage_layout()
                    .address(ordinal)
                    .expect("selected physical owner"),
                bindings.as_slice(),
            )
        })
    }

    /// Pinned canonical storage, including shared consumers.
    pub fn pinned_bindings(&self) -> &BTreeMap<ParameterStorageId, Vec<WeightBinding>> {
        &self.pinned
    }
    /// Retained exact placement, when selected.
    pub fn local_layout(&self) -> Option<&LocalModelLayout> {
        self.local_layout.as_ref()
    }

    /// Selected canonical unit addresses, independent of dense local storage order.
    pub fn unit_layout(&self) -> &ExecutionUnitLayout {
        self.selected_units
            .as_ref()
            .unwrap_or_else(|| self.declaration.parameters().unit_layout())
    }

    /// Selected logical consumers and their canonical physical owners.
    pub fn invocations(&self) -> &crate::ExecutionUnitAddressMap {
        &self.invocations
    }

    /// Restricts authority to one selected invocation's actual physical inputs.
    pub fn unit_source(
        &self,
        address: ExecutionUnitAddress,
    ) -> Result<SharedCheckpointSource, ComponentPreparationError> {
        let ordinal = self
            .unit_layout()
            .ordinal(address.group(), address.index())
            .ok_or(crate::ExecutionUnitLayoutError::UnknownSelectedAddress {
                group: address.group(),
                index: address.index(),
            })?;
        let storage = self
            .invocations
            .resolve(ordinal, address)?
            .storage_location();
        restrict_bindings(
            Arc::clone(&self.source),
            &self.id,
            self.units[storage.ordinal()].iter(),
        )
    }

    /// Exact authority for the retained pinned aggregate, excluding leased units.
    pub fn pinned_source(&self) -> Result<SharedCheckpointSource, ComponentPreparationError> {
        restrict_bindings(
            Arc::clone(&self.source),
            &self.id,
            self.pinned.values().flatten(),
        )
    }

    /// Applies native immutable storage preferences to the same cold bindings.
    /// This changes neither recipes, source authority, placement nor physical bytes.
    /// Modules supply preferences only; they do not select checkpoint parameters.
    pub fn with_storage_preferences<B, S, A>(
        mut self,
        architecture: &A,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, A::Error>
    where
        B: eredu_nn::NeuralBackend,
        S: crate::RuntimeState<B>,
        A: crate::LayeredArchitecture<B, S>,
        A::Error: From<ComponentPreparationError>,
    {
        if architecture.execution_graph()? != *self.declaration.parameters().graph() {
            return Err(ComponentPreparationError::ArchitectureMismatch(self.id).into());
        }
        let layout_error = |error| match error {
            crate::ArchitectureExecutionLayoutError::Architecture(error) => error,
            crate::ArchitectureExecutionLayoutError::Layout(error) => {
                A::Error::from(ComponentPreparationError::Units(error))
            }
        };
        let layout = architecture.execution_unit_layout().map_err(layout_error)?;
        if &layout != self.declaration.parameters().unit_layout() {
            return Err(ComponentPreparationError::ArchitectureMismatch(self.id).into());
        }
        let invocations = architecture
            .execution_unit_storage_map()
            .map_err(layout_error)?;
        if &invocations != self.declaration.parameters().invocations() {
            return Err(ComponentPreparationError::ArchitectureMismatch(self.id).into());
        }
        for bindings in self.pinned.values_mut() {
            *bindings = crate::apply_parameter_storage_layouts(
                architecture.static_modules(),
                std::mem::take(bindings),
            );
        }
        for (ordinal, bindings) in self.units.iter_mut().enumerate() {
            let storage = self
                .invocations
                .storage_location(ordinal)
                .expect("selected physical owner");
            let unit = architecture.build_storage_unit(storage, context)?;
            *bindings = crate::apply_parameter_storage_layouts(&unit, std::mem::take(bindings));
        }
        Ok(self)
    }

    /// Projects parallel placement from this preparation's retained declaration.
    /// A retained selection must match exactly; no recipe is transformed here.
    /// Family topology restrictions and additional geometry checks remain separate.
    pub fn parallel_layout(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<LocalModelLayout, ComponentPreparationError> {
        let layout =
            crate::derive_partitioned_local_layout(self.declaration.parameters(), topology)
                .map_err(|reason| ComponentPreparationError::ParallelLayout {
                    component: self.id.clone(),
                    reason,
                })?;
        if self
            .local_layout
            .as_ref()
            .is_some_and(|retained| retained != &layout)
        {
            return Err(ComponentPreparationError::SelectionMismatch(
                self.id.clone(),
            ));
        }
        Ok(layout)
    }

    /// Admits a component's tensor cohort before deriving its exact placement.
    /// Pipeline, expert and data ownership require an enclosing partition selection.
    /// Family geometry validation and consuming recipe placement remain separate.
    pub fn tensor_parallel_layout(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<LocalModelLayout, ComponentPreparationError> {
        if topology.pipeline_parallel_size() != 1
            || topology.expert_parallel_size() != 1
            || topology.data_parallel_size() != 1
        {
            return Err(ComponentPreparationError::TensorOnlyTopology {
                component: self.id.clone(),
                topology,
            });
        }
        self.parallel_layout(topology)
    }

    /// Applies placement once. Reusing the same selection retains the recipes;
    /// a different selection fails before any transform is applied.
    pub fn select(mut self, layout: &LocalModelLayout) -> Result<Self, ComponentPreparationError> {
        if let Some(selected) = &self.local_layout {
            return if selected == layout {
                Ok(self)
            } else {
                Err(ComponentPreparationError::SelectionMismatch(self.id))
            };
        }
        for bindings in self.units.iter_mut().chain(self.pinned.values_mut()) {
            *bindings = place_resources(std::mem::take(bindings), self.source.as_ref(), layout)?;
        }
        self.local_layout = Some(layout.clone());
        Ok(self)
    }

    /// Applies family-owned recipe policy without changing canonical targets,
    /// shapes, placement or source authority. The operation consumes its input.
    pub fn transform_units<E>(
        mut self,
        transform: impl FnOnce(&SharedCheckpointSource, &mut [Vec<WeightBinding>]) -> Result<(), E>,
    ) -> Result<Self, E>
    where
        E: From<ComponentPreparationError>,
    {
        let mut transformed = self.units.clone();
        transform(&self.source, &mut transformed)?;
        for (original, selected) in self.units.iter().zip(&transformed) {
            if original.len() != selected.len() {
                return Err(ComponentPreparationError::InvalidProjection {
                    component: self.id,
                    target: "unit resource count".into(),
                }
                .into());
            }
            for (original, selected) in original.iter().zip(selected) {
                if selected.name() != original.name()
                    || selected.logical_target() != original.logical_target()
                    || selected.alias_of() != original.alias_of()
                    || selected.checkpoint_keys() != original.checkpoint_keys()
                    || selected.storage_layout() != original.storage_layout()
                    || selected.quantization_companions() != original.quantization_companions()
                {
                    return Err(ComponentPreparationError::InvalidProjection {
                        component: self.id,
                        target: original.name().into(),
                    }
                    .into());
                }
                if selected.is_alias() {
                    continue;
                }
                let metadata = selected
                    .source_recipe()
                    .infer(self.source.as_ref())
                    .map_err(ComponentPreparationError::from)?;
                let previous = original
                    .source_recipe()
                    .infer(self.source.as_ref())
                    .map_err(ComponentPreparationError::from)?;
                if metadata.shape != previous.shape
                    || metadata.byte_len() != selected.expected_bytes()
                {
                    return Err(ComponentPreparationError::InvalidProjection {
                        component: self.id,
                        target: original.name().into(),
                    }
                    .into());
                }
            }
        }
        for (original, transformed) in self.units.iter().zip(&mut transformed) {
            let plan = WeightBindingPlan::new(original)
                .map_err(RecipeBindingError::from)
                .map_err(ComponentPreparationError::from)?;
            retain_aliases(plan.alias_indices(), transformed)?;
        }
        self.units = transformed;
        Ok(self)
    }

    /// Selects canonical targets by identity, retaining transformed bindings and
    /// restricting source authority again. Placement is projected, never reapplied.
    pub fn project<P: ComponentPreparationDeclaration>(
        self,
        id: impl Into<String>,
        declaration: P,
    ) -> Result<PreparedComponent<P>, ComponentPreparationError> {
        let id = id.into();
        let description = declaration.parameters();
        let layout = description.unit_layout();
        let retained = self.unit_layout();
        for group in 0..layout.group_count() {
            let name = layout.group_id(group).expect("declared group");
            let original =
                (0..retained.group_count()).find(|&index| retained.group_id(index) == Some(name));
            for ordinal in layout.group_range(group).expect("declared group") {
                let address = layout.address(ordinal).expect("declared invocation");
                if original.is_none_or(|index| retained.ordinal(index, address.index()).is_none()) {
                    return Err(ComponentPreparationError::InvalidProjection {
                        component: id,
                        target: format!("invocation {}:{}", name.as_str(), address.index()),
                    });
                }
            }
        }
        let members = self
            .declaration
            .parameters()
            .groups()
            .iter()
            .flat_map(|group| {
                group
                    .members()
                    .iter()
                    .map(move |member| (member.target(), (group.owner(), member)))
            })
            .collect::<BTreeMap<_, _>>();
        for group in description.groups() {
            for member in group.members() {
                if members
                    .get(member.target())
                    .is_none_or(|(owner, original)| *owner != group.owner() || *original != member)
                {
                    return Err(ComponentPreparationError::InvalidProjection {
                        component: id,
                        target: member.target().into(),
                    });
                }
            }
        }
        let mut bindings = self
            .units
            .into_iter()
            .flatten()
            .chain(self.pinned.into_values().flatten())
            .map(|binding| {
                (
                    binding
                        .logical_target()
                        .unwrap_or(binding.name())
                        .to_owned(),
                    binding,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut units = vec![Vec::new(); description.invocations().storage_layout().len()];
        let mut pinned = BTreeMap::<ParameterStorageId, Vec<WeightBinding>>::new();
        let mut local_layout = self
            .local_layout
            .as_ref()
            .map(|_| LocalModelLayout::default());
        for group in description.groups() {
            let owner = group.owner();
            let destination = if let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = owner
            {
                let group = description
                    .graph()
                    .group_index(group.as_str())
                    .expect("validated group");
                let ordinal = description
                    .invocations()
                    .storage_layout()
                    .ordinal(group, *global_unit)
                    .expect("validated unit");
                &mut units[ordinal]
            } else {
                pinned
                    .entry(ParameterStorageId::from_owner(owner))
                    .or_default()
            };
            for member in group.members() {
                let invalid = || ComponentPreparationError::InvalidProjection {
                    component: id.clone(),
                    target: member.target().into(),
                };
                destination.push(bindings.remove(member.target()).ok_or_else(invalid)?);
                if let Some(selected) = &mut local_layout {
                    let tensor = self
                        .local_layout
                        .as_ref()
                        .and_then(|layout| layout.tensor(member.target()))
                        .ok_or_else(invalid)?;
                    selected.insert(member.target().into(), tensor.clone());
                }
            }
        }
        for bindings in units.iter().chain(pinned.values()) {
            WeightBindingPlan::new(bindings).map_err(RecipeBindingError::from)?;
        }
        let source = restrict_bindings(
            self.source,
            &id,
            units.iter().flatten().chain(pinned.values().flatten()),
        )?;
        let invocations = description.invocations().clone();
        Ok(PreparedComponent {
            id,
            declaration,
            source,
            units,
            selected_units: None,
            invocations,
            pinned,
            local_layout,
        })
    }

    /// Restricts independently leased units by their global invocation addresses.
    /// Dense local ordinals never replace the selected global addresses. Retained
    /// tensor placement remains authoritative and is not transformed again.
    pub fn select_units(
        mut self,
        addresses: impl IntoIterator<Item = ExecutionUnitAddress>,
    ) -> Result<Self, ComponentPreparationError> {
        let addresses = addresses.into_iter().collect::<Vec<_>>();
        let selected = self.unit_layout().select(addresses.iter().copied())?;
        let invocations = self.invocations.select_invocations(addresses)?;
        let mut retained = self.units.into_iter().map(Some).collect::<Vec<_>>();
        self.units = (0..invocations.storage_layout().len())
            .map(|ordinal| {
                let address = invocations
                    .storage_layout()
                    .address(ordinal)
                    .expect("selected physical owner");
                let original = self
                    .invocations
                    .storage_layout()
                    .ordinal(address.group(), address.index())
                    .expect("validated subset");
                retained[original].take().expect("unique selected address")
            })
            .collect();
        self.selected_units = Some(selected);
        self.invocations = invocations;
        self.source = restrict_bindings(
            self.source,
            &self.id,
            self.units
                .iter()
                .flatten()
                .chain(self.pinned.values().flatten()),
        )?;
        Ok(self)
    }

    /// Transfers exact retained placement, addresses, sources and canonical resources.
    pub fn into_parts(
        self,
    ) -> (
        Option<LocalModelLayout>,
        crate::ExecutionUnitAddressMap,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
        BTreeMap<ParameterStorageId, Vec<WeightBinding>>,
    ) {
        (
            self.local_layout,
            self.invocations,
            self.source,
            self.units,
            self.pinned,
        )
    }

    /// Transfers unit-only authority with explicit canonical physical owner addresses.
    pub fn into_addressed_unit_parts(
        self,
    ) -> Result<
        (
            D,
            Option<LocalModelLayout>,
            crate::ExecutionUnitAddressMap,
            SharedCheckpointSource,
            Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
        ),
        ComponentPreparationError,
    > {
        let (declaration, local, layout, source, units) = self.into_declared_unit_parts()?;
        let units = units
            .into_iter()
            .enumerate()
            .map(|(ordinal, bindings)| {
                (
                    layout
                        .storage_layout()
                        .address(ordinal)
                        .expect("selected physical owner"),
                    bindings,
                )
            })
            .collect();
        Ok((declaration, local, layout, source, units))
    }

    /// Moves the original typed declaration with its selected unit resources.
    /// Global semantic ownership stays in the declaration; the returned map
    /// retains selected consumers and physical owners without rebuilding facts.
    pub fn into_declared_unit_parts(
        self,
    ) -> Result<
        (
            D,
            Option<LocalModelLayout>,
            crate::ExecutionUnitAddressMap,
            SharedCheckpointSource,
            Vec<Vec<WeightBinding>>,
        ),
        ComponentPreparationError,
    > {
        if !self.pinned.is_empty() {
            return Err(ComponentPreparationError::PinnedResources(self.id));
        }
        Ok((
            self.declaration,
            self.local_layout,
            self.invocations,
            self.source,
            self.units,
        ))
    }
}

fn restrict_bindings<'a>(
    source: SharedCheckpointSource,
    id: &str,
    bindings: impl Iterator<Item = &'a WeightBinding>,
) -> Result<SharedCheckpointSource, ComponentPreparationError> {
    let inputs = bindings
        .flat_map(WeightBinding::checkpoint_keys)
        .map(str::to_owned)
        .collect();
    Ok(Arc::new(RestrictedCheckpointSource::including(
        source, id, inputs,
    )?))
}

// Placement and precision change physical byte counts; aliases keep referring to
// the same canonical owner and adopt its selected physical size.
// Both callers preserve binding order; plan indices resolve alias chains to physical owners.
fn retain_aliases(
    aliases: impl IntoIterator<Item = (usize, usize)>,
    selected: &mut [WeightBinding],
) -> Result<(), ComponentPreparationError> {
    for (alias_index, owner_index) in aliases {
        let bytes = selected[owner_index].expected_bytes();
        let binding = &mut selected[alias_index];
        let mut alias = WeightBinding::alias(
            binding.name(),
            binding.alias_of().expect("validated alias"),
            bytes,
        )
        .map_err(RecipeBindingError::from)?
        .with_storage_layout(binding.storage_layout());
        if let Some(target) = binding.logical_target() {
            alias = alias
                .with_logical_target(target)
                .map_err(RecipeBindingError::from)?;
        }
        *binding = alias;
    }
    WeightBindingPlan::new(selected).map_err(RecipeBindingError::from)?;
    Ok(())
}

fn place_resources(
    bindings: Vec<WeightBinding>,
    source: &dyn eredu_checkpoint::store::CheckpointSource,
    layout: &LocalModelLayout,
) -> Result<Vec<WeightBinding>, ComponentPreparationError> {
    let plan = WeightBindingPlan::new(&bindings).map_err(RecipeBindingError::from)?;
    for (alias, owner) in plan.aliases() {
        let alias_layout = alias
            .logical_target()
            .and_then(|target| layout.tensor(target));
        let owner_layout = owner
            .logical_target()
            .and_then(|target| layout.tensor(target));
        if alias_layout.zip(owner_layout).is_none_or(|(alias, owner)| {
            alias.global_shape() != owner.global_shape()
                || alias.local_shape() != owner.local_shape()
                || alias.placement() != owner.placement()
                || alias.additional_placements() != owner.additional_placements()
        }) {
            return Err(ComponentPreparationError::AliasPlacement {
                alias: alias.name().into(),
                owner: owner.name().into(),
            });
        }
    }
    let aliases = plan.alias_indices().collect::<Vec<_>>();
    let mut selected = place_weight_bindings(bindings, source, layout)?;
    retain_aliases(aliases, &mut selected)?;
    Ok(selected)
}
