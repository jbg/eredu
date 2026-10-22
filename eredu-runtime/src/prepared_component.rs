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
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// Failure preparing exact component resources without allocating native values.
#[derive(Debug, thiserror::Error)]
pub enum ComponentPreparationError {
    /// Selected unit identities are not a canonical subset.
    #[error(transparent)]
    Units(#[from] ExecutionUnitLayoutError),
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

/// Consuming exact bindings with restricted source authority and retained placement.
///
/// Family code supplies admitted recipes, including precision policy and special
/// transforms. This container validates global geometry, groups physical ownership,
/// restricts source access, and applies rank-local placement at most once.
pub struct PreparedComponent {
    id: String,
    description: ArchitectureParameterDescription,
    source: SharedCheckpointSource,
    units: Vec<Vec<WeightBinding>>,
    unit_layout: ExecutionUnitLayout,
    pinned: BTreeMap<ParameterStorageId, Vec<WeightBinding>>,
    local_layout: Option<LocalModelLayout>,
}

impl PreparedComponent {
    /// Resolves canonical targets using family-owned admitted recipe policy.
    /// No checkpoint payload is read.
    pub fn prepare<E, R>(
        id: impl Into<String>,
        description: ArchitectureParameterDescription,
        source: SharedCheckpointSource,
        mut resolve: impl FnMut(&ParameterMemberSpec) -> Result<R, E>,
    ) -> Result<Self, E>
    where
        E: From<ComponentPreparationError>,
        R: Into<ComponentRecipe>,
    {
        let id = id.into();
        let mut recipes =
            BTreeMap::<ParameterStorageId, BTreeMap<String, DerivedWeightRecipe>>::new();
        let mut aliases = BTreeMap::<ParameterStorageId, Vec<RecipeAlias>>::new();
        for owned in description.groups() {
            let owner = ParameterStorageId(owned.owner().clone());
            for member in owned.members() {
                match resolve(member)?.into() {
                    ComponentRecipe::Physical { recipe, conversion } => {
                        let recipe = match conversion {
                            Some(conversion) => {
                                conversion.apply(recipe, source.as_ref(), &id, member.target())?
                            }
                            None => recipe,
                        };
                        recipes
                            .entry(owner.clone())
                            .or_default()
                            .insert(member.target().into(), recipe);
                    }
                    ComponentRecipe::Alias(target) => {
                        aliases
                            .entry(owner.clone())
                            .or_default()
                            .push(RecipeAlias::new(member.target(), target));
                        recipes.entry(owner.clone()).or_default();
                    }
                }
            }
        }
        let resources = recipes
            .into_iter()
            .map(|(owner, outputs)| {
                let aliases = aliases.remove(&owner).unwrap_or_default();
                (owner, ParameterResourceRecipes { outputs, aliases })
            })
            .collect();
        let (source, resources) = prepare_resource_bindings(source, &id, resources)
            .map_err(ComponentPreparationError::from)?;
        let mut units = vec![Vec::new(); description.unit_layout().len()];
        let mut pinned = BTreeMap::new();
        for (owner, bindings) in resources {
            let plan = WeightBindingPlan::new(&bindings)
                .map_err(RecipeBindingError::from)
                .map_err(ComponentPreparationError::from)?;
            let resolved = plan
                .owners()
                .map(|binding| (binding.name(), binding))
                .chain(plan.aliases().map(|(alias, owner)| (alias.name(), owner)))
                .collect::<BTreeMap<_, _>>();
            let mut targets = BTreeSet::new();
            for group in description
                .groups()
                .iter()
                .filter(|group| group.owner() == &owner.0)
            {
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
                    targets.insert(member.target().to_owned());
                }
            }
            let bindings = bindings
                .into_iter()
                .map(|binding| {
                    let target = binding.name().to_owned();
                    if !targets.contains(&target) {
                        return Err(ComponentPreparationError::Geometry {
                            component: id.clone(),
                            target,
                        });
                    }
                    binding
                        .with_logical_target(target)
                        .map_err(RecipeBindingError::from)
                        .map_err(ComponentPreparationError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if let ParameterGroupOwner::ExecutionUnit {
                group, global_unit, ..
            } = &owner.0
            {
                let group = description
                    .graph()
                    .group_index(group.as_str())
                    .expect("validated parameter owner");
                let ordinal = description
                    .unit_layout()
                    .ordinal(group, *global_unit)
                    .expect("validated execution unit");
                units[ordinal] = bindings;
            } else {
                pinned.insert(owner, bindings);
            }
        }
        let unit_layout = description.unit_layout().clone();
        Ok(Self {
            id,
            description,
            source,
            units,
            unit_layout,
            pinned,
            local_layout: None,
        })
    }

    /// Validated canonical ownership used by this exact preparation.
    pub fn description(&self) -> &ArchitectureParameterDescription {
        &self.description
    }
    /// Exact retained source authority.
    pub fn source(&self) -> &SharedCheckpointSource {
        &self.source
    }
    /// Independently leased units in canonical layout order.
    pub fn unit_bindings(&self) -> &[Vec<WeightBinding>] {
        &self.units
    }
    /// Selected logical addresses paired with their independently leased bindings.
    /// Dense storage ordinals never replace group or global unit identities.
    pub fn addressed_unit_bindings(
        &self,
    ) -> impl ExactSizeIterator<Item = (ExecutionUnitAddress, &[WeightBinding])> {
        self.units.iter().enumerate().map(|(ordinal, bindings)| {
            (
                self.unit_layout.address(ordinal).expect("selected unit"),
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
        &self.unit_layout
    }

    /// Restricts authority to one selected invocation's actual physical inputs.
    pub fn unit_source(
        &self,
        address: ExecutionUnitAddress,
    ) -> Result<SharedCheckpointSource, ComponentPreparationError> {
        let ordinal = self
            .unit_layout
            .ordinal(address.group(), address.index())
            .ok_or(crate::ExecutionUnitLayoutError::UnknownSelectedAddress {
                group: address.group(),
                index: address.index(),
            })?;
        self.restrict_bindings(self.units[ordinal].iter())
    }

    /// Exact authority for the retained pinned aggregate, excluding leased units.
    pub fn pinned_source(&self) -> Result<SharedCheckpointSource, ComponentPreparationError> {
        self.restrict_bindings(self.pinned.values().flatten())
    }

    fn restrict_bindings<'a>(
        &self,
        bindings: impl Iterator<Item = &'a WeightBinding>,
    ) -> Result<SharedCheckpointSource, ComponentPreparationError> {
        let inputs = bindings
            .flat_map(|binding| {
                binding
                    .checkpoint_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(Arc::new(RestrictedCheckpointSource::including(
            self.source.clone(),
            &self.id,
            inputs,
        )?))
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
        if architecture.execution_graph()? != *self.description.graph() {
            return Err(ComponentPreparationError::ArchitectureMismatch(self.id).into());
        }
        for group in 0..self.description.graph().groups().len() {
            if architecture.group_unit_count(group)?
                != self
                    .description
                    .unit_layout()
                    .group_range(group)
                    .expect("declared group")
                    .len()
            {
                return Err(ComponentPreparationError::ArchitectureMismatch(self.id).into());
            }
        }
        for bindings in self.pinned.values_mut() {
            *bindings = crate::apply_parameter_storage_layouts(
                architecture.static_modules(),
                std::mem::take(bindings),
            );
        }
        for (ordinal, bindings) in self.units.iter_mut().enumerate() {
            let address = self.unit_layout.address(ordinal).expect("selected unit");
            let unit = architecture.build_unit(address.group(), address.index(), context)?;
            *bindings = crate::apply_parameter_storage_layouts(&unit, std::mem::take(bindings));
        }
        Ok(self)
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
        self.units = self
            .units
            .into_iter()
            .map(|bindings| place_resources(bindings, self.source.as_ref(), layout))
            .collect::<Result<_, _>>()?;
        self.pinned = self
            .pinned
            .into_iter()
            .map(|(owner, bindings)| {
                Ok((
                    owner,
                    place_resources(bindings, self.source.as_ref(), layout)?,
                ))
            })
            .collect::<Result<_, ComponentPreparationError>>()?;
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
            retain_aliases(original, transformed)?;
        }
        self.units = transformed;
        Ok(self)
    }

    /// Selects canonical targets by identity, retaining transformed bindings and
    /// restricting source authority again. Placement is projected, never reapplied.
    pub fn project(
        self,
        id: impl Into<String>,
        description: ArchitectureParameterDescription,
    ) -> Result<Self, ComponentPreparationError> {
        let id = id.into();
        let members = self
            .description
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
        let mut units = vec![Vec::new(); description.unit_layout().len()];
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
                    .unit_layout()
                    .ordinal(group, *global_unit)
                    .expect("validated unit");
                &mut units[ordinal]
            } else {
                pinned.entry(ParameterStorageId(owner.clone())).or_default()
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
        let inputs = units
            .iter()
            .flatten()
            .chain(pinned.values().flatten())
            .flat_map(|binding| {
                binding
                    .checkpoint_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        let source = Arc::new(RestrictedCheckpointSource::including(
            self.source,
            &id,
            inputs,
        )?);
        let unit_layout = description.unit_layout().clone();
        Ok(Self {
            id,
            description,
            source,
            units,
            unit_layout,
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
        let selected = self.unit_layout.select(addresses)?;
        let mut retained = self.units.into_iter().map(Some).collect::<Vec<_>>();
        self.units = (0..selected.len())
            .map(|ordinal| {
                let address = selected.address(ordinal).expect("selected address");
                let original = self
                    .unit_layout
                    .ordinal(address.group(), address.index())
                    .expect("validated subset");
                retained[original].take().expect("unique selected address")
            })
            .collect();
        self.unit_layout = selected;
        let inputs = self
            .units
            .iter()
            .flatten()
            .chain(self.pinned.values().flatten())
            .flat_map(|binding| {
                binding
                    .checkpoint_keys()
                    .into_iter()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect();
        self.source = Arc::new(RestrictedCheckpointSource::including(
            self.source,
            &self.id,
            inputs,
        )?);
        Ok(self)
    }

    /// Transfers exact retained placement, addresses, sources and canonical resources.
    pub fn into_parts(
        self,
    ) -> (
        Option<LocalModelLayout>,
        ExecutionUnitLayout,
        SharedCheckpointSource,
        Vec<Vec<WeightBinding>>,
        BTreeMap<ParameterStorageId, Vec<WeightBinding>>,
    ) {
        (
            self.local_layout,
            self.unit_layout,
            self.source,
            self.units,
            self.pinned,
        )
    }

    /// Transfers unit-only authority with explicit canonical invocation addresses.
    pub fn into_addressed_unit_parts(
        self,
    ) -> Result<
        (
            Option<LocalModelLayout>,
            ExecutionUnitLayout,
            SharedCheckpointSource,
            Vec<(ExecutionUnitAddress, Vec<WeightBinding>)>,
        ),
        ComponentPreparationError,
    > {
        let (local, layout, source, units) = self.into_unit_parts()?;
        let units = units
            .into_iter()
            .enumerate()
            .map(|(ordinal, bindings)| (layout.address(ordinal).expect("selected unit"), bindings))
            .collect();
        Ok((local, layout, source, units))
    }

    /// Transfers exact unit-only authority to existing materialization drivers.
    pub fn into_unit_parts(
        self,
    ) -> Result<
        (
            Option<LocalModelLayout>,
            ExecutionUnitLayout,
            SharedCheckpointSource,
            Vec<Vec<WeightBinding>>,
        ),
        ComponentPreparationError,
    > {
        if !self.pinned.is_empty() {
            return Err(ComponentPreparationError::PinnedResources(self.id));
        }
        Ok((self.local_layout, self.unit_layout, self.source, self.units))
    }
}

// Placement and precision change physical byte counts; aliases keep referring to
// the same canonical owner and adopt its selected physical size.
fn retain_aliases(
    original: &[WeightBinding],
    selected: &mut [WeightBinding],
) -> Result<(), ComponentPreparationError> {
    let plan = WeightBindingPlan::new(original).map_err(RecipeBindingError::from)?;
    let owners = plan
        .aliases()
        .map(|(alias, owner)| (alias.name().to_owned(), owner.name().to_owned()))
        .collect::<BTreeMap<_, _>>();
    let bytes = selected
        .iter()
        .map(|binding| (binding.name().to_owned(), binding.expected_bytes()))
        .collect::<BTreeMap<_, _>>();
    for binding in selected.iter_mut() {
        if let Some(owner) = owners.get(binding.name()) {
            let mut alias = WeightBinding::alias(
                binding.name(),
                binding.alias_of().expect("validated alias"),
                bytes[owner],
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
    let mut selected = place_weight_bindings(bindings.clone(), source, layout)?;
    retain_aliases(&bindings, &mut selected)?;
    Ok(selected)
}
