//! Backend-neutral topology and checkpoint recipes for independent expert residency.

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_checkpoint::{recipe::RecipeCatalog, store::TensorSelection};
use eredu_core::{
    balanced_contiguous_range, CollectiveGroupDescriptor, CollectiveGroupId, ParallelAxis,
    ParallelRankTopology,
};
use eredu_runtime::{
    AddressableGatedProductBank, CollectiveBackend, ExecutionGroupId, ParameterBankKey,
};

/// Complete architecture-derived ownership and rank-local bank construction plan.
///
/// The plan is deliberately independent of a concrete collective runtime. It
/// fixes the global-to-owner mapping once and carries the architecture's exact
/// rank-local construction specification for every routed execution unit.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRealizationPlan<S> {
    global_expert_count: usize,
    expert_parallel_size: usize,
    expert_parallel_rank: usize,
    owners: Vec<usize>,
    local_global_group_indices: Vec<usize>,
    collective_members: Vec<usize>,
    collective_local_rank: usize,
    unit_specs: BTreeMap<(ExecutionGroupId, usize), S>,
}

impl<S> ExpertRealizationPlan<S> {
    /// Creates a balanced contiguous realization for one architecture rank.
    pub fn balanced(
        global_expert_count: usize,
        topology: ParallelRankTopology,
        unit_specs: BTreeMap<(ExecutionGroupId, usize), S>,
    ) -> Result<Self, ExpertRealizationPlanError> {
        if global_expert_count == 0 {
            return Err(ExpertRealizationPlanError::EmptyExpertBank);
        }
        if unit_specs.is_empty() {
            return Err(ExpertRealizationPlanError::EmptyUnitSchedule);
        }
        let mut owners = vec![0; global_expert_count];
        for owner in 0..topology.expert_parallel_size() {
            let range = balanced_contiguous_range(
                global_expert_count,
                topology.expert_parallel_size(),
                owner,
                false,
            )
            .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
            owners[range].fill(owner);
        }
        let local = balanced_contiguous_range(
            global_expert_count,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
        let collective = topology
            .subgroup(ParallelAxis::Expert)
            .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
        Ok(Self {
            global_expert_count,
            expert_parallel_size: topology.expert_parallel_size(),
            expert_parallel_rank: topology.expert_parallel_rank(),
            owners,
            local_global_group_indices: local.collect(),
            collective_members: collective.global_ranks().to_vec(),
            collective_local_rank: collective.rank(),
            unit_specs,
        })
    }

    /// Returns the checkpoint-global routed expert count used by preflight.
    pub const fn global_expert_count(&self) -> usize {
        self.global_expert_count
    }

    /// Returns the expert-axis rank count used to derive ownership.
    pub const fn expert_parallel_size(&self) -> usize {
        self.expert_parallel_size
    }

    /// Returns this rank's coordinate on the expert axis.
    pub const fn expert_parallel_rank(&self) -> usize {
        self.expert_parallel_rank
    }

    /// Returns one owner rank for every checkpoint-global expert identity.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }

    /// Returns this rank's global expert identities in owner-local order.
    pub fn local_global_group_indices(&self) -> &[usize] {
        &self.local_global_group_indices
    }

    /// Returns the exact rank-local bank specification for an execution unit.
    pub fn unit_spec(&self, owner_group: &str, owner_unit: usize) -> Option<&S> {
        self.unit_specs
            .iter()
            .find(|((group, unit), _)| group.as_str() == owner_group && *unit == owner_unit)
            .map(|(_, spec)| spec)
    }

    /// Returns whether the plan declares any routed unit in an execution group.
    pub fn has_routed_units_in_group(&self, owner_group: &str) -> bool {
        self.unit_specs
            .keys()
            .any(|(group, _)| group.as_str() == owner_group)
    }

    /// Returns every routed execution unit and its rank-local bank specification.
    pub fn unit_specs(&self) -> &BTreeMap<(ExecutionGroupId, usize), S> {
        &self.unit_specs
    }

    pub(crate) fn try_map_unit_specs<T>(
        self,
        mut map: impl FnMut(S) -> Result<T, String>,
    ) -> Result<ExpertRealizationPlan<T>, String> {
        let unit_specs = self
            .unit_specs
            .into_iter()
            .map(|(address, spec)| map(spec).map(|mapped| (address, mapped)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(ExpertRealizationPlan {
            global_expert_count: self.global_expert_count,
            expert_parallel_size: self.expert_parallel_size,
            expert_parallel_rank: self.expert_parallel_rank,
            owners: self.owners,
            local_global_group_indices: self.local_global_group_indices,
            collective_members: self.collective_members,
            collective_local_rank: self.collective_local_rank,
            unit_specs,
        })
    }

    /// Translates the architecture's semantic route axis to an opaque generic group.
    pub fn collective_group(
        &self,
        id: CollectiveGroupId,
    ) -> Result<CollectiveGroupDescriptor, ExpertRealizationPlanError> {
        CollectiveGroupDescriptor::new(
            id,
            self.collective_members.clone(),
            self.collective_local_rank,
        )
        .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))
    }
}

/// Failure while translating and executing one routed plan through generic mechanisms.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedMechanismExecutionError {
    /// The architecture plan did not contain the requested local bank member or unit.
    #[error("invalid routed mechanism plan: {0}")]
    InvalidPlan(String),
    /// A mechanism-only bank or collective operation failed.
    #[error("routed mechanism execution failed: {0}")]
    Mechanism(String),
}

/// Executes an architecture-owned route through generic bank, grouped, and collective mechanisms.
#[allow(clippy::too_many_arguments)]
pub fn execute_routed_gated_product<B, P>(
    plan: &ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    owner_group: &str,
    owner_unit: usize,
    local_bank_member: usize,
    input: &B::Tensor,
    routes: &eredu_nn::GroupSelection<B::Tensor>,
    bank: &mut P,
    collective: &CollectiveGroupDescriptor,
    group: &B::Group,
    executor: &B::Executor,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<B::Tensor, RoutedMechanismExecutionError>
where
    B: eredu_nn::GroupedNeuralBackend + CollectiveBackend,
    P: AddressableGatedProductBank<B>,
    P::Error: std::fmt::Display,
    B::CollectiveError: std::fmt::Display,
{
    let expected = plan
        .collective_group(collective.id())
        .map_err(|error| RoutedMechanismExecutionError::InvalidPlan(error.to_string()))?;
    if &expected != collective {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "collective membership does not match the architecture route plan".into(),
        ));
    }
    let global_member = plan
        .local_global_group_indices()
        .get(local_bank_member)
        .copied()
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(format!(
                "local bank member {local_bank_member} is outside the selected bank"
            ))
        })?;
    let spec = plan.unit_spec(owner_group, owner_unit).ok_or_else(|| {
        RoutedMechanismExecutionError::InvalidPlan(format!(
            "execution unit {owner_group:?}/{owner_unit} has no grouped bank"
        ))
    })?;
    let key = ParameterBankKey::new(owner_unit, global_member);
    let groups = bank
        .acquire(key, spec, context)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let output =
        eredu_nn::GroupedGatedProductOperator::forward_grouped(groups, input, routes, context)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    if collective.members().len() > 1 {
        B::all_to_all(output, group, executor)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))
    } else {
        Ok(output)
    }
}

/// Invalid architecture-derived expert realization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExpertRealizationPlanError {
    /// The architecture declared no routed experts.
    #[error("expert realization requires at least one routed expert")]
    EmptyExpertBank,
    /// The architecture declared no routed execution units.
    #[error("expert realization requires at least one routed execution unit")]
    EmptyUnitSchedule,
    /// The requested rank topology cannot own a non-empty expert partition.
    #[error("invalid expert realization topology: {0}")]
    InvalidTopology(String),
}

/// Placement of one expert relative to an expert-parallel axis.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertResidencyDistribution {
    /// Assign the expert by its global router identity across expert ranks.
    ExpertParallel,
    /// Materialize the expert on every rank that owns its execution unit.
    Replicated,
}

/// Selects architecture-canonical expert outputs for one rank's global IDs.
///
/// The expert axis is part of the architecture's canonical parameter geometry.
/// Applying the selection to derived outputs lets the checkpoint recipe layer
/// push it through fused, stacked, or transposed physical layouts without a
/// backend recovering storage geometry.
pub(crate) fn select_rank_local_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    global_experts: usize,
    expert_axis: usize,
    group_indices: &[usize],
    outputs: impl IntoIterator<Item = (String, DerivedWeightRecipe)>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if group_indices.is_empty() {
        return Err("rank-local expert recipes require at least one expert".into());
    }
    let mut unique = BTreeSet::new();
    for &expert in group_indices {
        if expert >= global_experts {
            return Err(format!(
                "rank-local expert {expert} is outside {global_experts} experts"
            ));
        }
        if !unique.insert(expert) {
            return Err(format!(
                "rank-local expert recipe contains duplicate expert {expert}"
            ));
        }
    }
    let selection = TensorSelection::Indices {
        axis: expert_axis,
        indices: group_indices.to_vec(),
    };
    outputs
        .into_iter()
        .map(|(target, recipe)| {
            recipe
                .select_bounded(catalog, selection.clone())
                .map(|recipe| (target, recipe))
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// One architecture-logical parameter in an independently resident expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertParameterRecipe {
    binding_name: String,
    logical_target: String,
    recipe: DerivedWeightRecipe,
    role: ExpertParameterRole,
    metadata: Option<eredu_checkpoint::recipe::RecipeMetadata>,
}

/// Quantization semantics of one independently resident expert parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ExpertParameterRole {
    /// Preserve this binding exactly as declared by the architecture.
    Preserved,
    /// Quantize this projection and publish companions under these exact local names.
    QuantizableProjection {
        /// Local binding name for packed quantization scales.
        scales_binding: String,
        /// Local binding name for packed affine biases, when the format uses them.
        biases_binding: String,
    },
}

impl ExpertParameterRole {
    /// Declares a projection eligible for load-time quantization.
    pub fn quantizable_projection(
        scales_binding: impl Into<String>,
        biases_binding: impl Into<String>,
    ) -> Self {
        Self::QuantizableProjection {
            scales_binding: scales_binding.into(),
            biases_binding: biases_binding.into(),
        }
    }
}

impl ExpertParameterRecipe {
    /// Creates one exact local binding and its architecture-logical destination.
    pub fn new(
        binding_name: impl Into<String>,
        logical_target: impl Into<String>,
        recipe: DerivedWeightRecipe,
        role: ExpertParameterRole,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let binding_name = binding_name.into();
        if binding_name.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyBindingName);
        }
        let logical_target = logical_target.into();
        if logical_target.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyLogicalTarget {
                binding: binding_name,
            });
        }
        if recipe.source_keys().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyRecipe {
                binding: binding_name,
            });
        }
        if let ExpertParameterRole::QuantizableProjection {
            scales_binding,
            biases_binding,
        } = &role
        {
            for companion in [scales_binding, biases_binding] {
                if companion.trim().is_empty() {
                    return Err(ExpertResidencyCatalogError::EmptyQuantizationCompanion {
                        binding: binding_name,
                    });
                }
                if companion == &binding_name {
                    return Err(
                        ExpertResidencyCatalogError::QuantizationCompanionCollision {
                            binding: binding_name,
                            companion: companion.clone(),
                        },
                    );
                }
            }
            if scales_binding == biases_binding {
                return Err(
                    ExpertResidencyCatalogError::QuantizationCompanionCollision {
                        binding: binding_name,
                        companion: scales_binding.clone(),
                    },
                );
            }
        }
        Ok(Self {
            binding_name,
            logical_target,
            recipe,
            role,
            metadata: None,
        })
    }

    /// Returns the stable name used by an acquired expert bank.
    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    /// Returns the exact architecture parameter destination.
    pub fn logical_target(&self) -> &str {
        &self.logical_target
    }

    /// Returns the checkpoint-derived recipe for this expert-local value.
    pub const fn recipe(&self) -> &DerivedWeightRecipe {
        &self.recipe
    }

    /// Returns the architecture-declared parameter and quantization semantics.
    pub const fn role(&self) -> &ExpertParameterRole {
        &self.role
    }

    /// Returns admission-time metadata for this expert-local recipe output.
    pub const fn metadata(&self) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        self.metadata.as_ref()
    }

    /// Consumes the declaration into a named handoff artifact.
    pub fn into_artifact(self) -> ExpertParameterArtifact {
        ExpertParameterArtifact {
            binding_name: Some(self.binding_name),
            logical_target: Some(self.logical_target),
            recipe: Some(self.recipe),
            role: Some(self.role),
        }
    }
}

/// Named consuming artifact for one expert-local parameter declaration.
pub struct ExpertParameterArtifact {
    binding_name: Option<String>,
    logical_target: Option<String>,
    recipe: Option<DerivedWeightRecipe>,
    role: Option<ExpertParameterRole>,
}

impl ExpertParameterArtifact {
    /// Takes the local binding name exactly once.
    pub fn take_binding_name(&mut self) -> String {
        self.binding_name
            .take()
            .expect("binding name already taken")
    }
    /// Takes the architecture-logical destination exactly once.
    pub fn take_logical_target(&mut self) -> String {
        self.logical_target
            .take()
            .expect("logical target already taken")
    }
    /// Takes the checkpoint-derived recipe exactly once.
    pub fn take_recipe(&mut self) -> DerivedWeightRecipe {
        self.recipe.take().expect("expert recipe already taken")
    }
    /// Takes parameter-role semantics exactly once.
    pub fn take_role(&mut self) -> ExpertParameterRole {
        self.role
            .take()
            .expect("expert parameter role already taken")
    }
}

/// One independently addressable expert and its owning execution unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertResidencyUnit {
    identity: ParameterBankKey,
    owner_group: ExecutionGroupId,
    owner_unit: usize,
    unit_path: String,
    distribution: ExpertResidencyDistribution,
    parameters: Vec<ExpertParameterRecipe>,
    byte_len: Option<u64>,
}

impl ExpertResidencyUnit {
    /// Creates one complete atomic expert unit.
    pub fn new(
        identity: ParameterBankKey,
        owner_group: ExecutionGroupId,
        owner_unit: usize,
        unit_path: impl Into<String>,
        distribution: ExpertResidencyDistribution,
        parameters: impl IntoIterator<Item = ExpertParameterRecipe>,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let unit_path = unit_path.into();
        if unit_path.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyUnitPath { identity });
        }
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        if parameters.is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyUnit { identity });
        }
        let mut names = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for parameter in &parameters {
            if !names.insert(parameter.binding_name.as_str()) {
                return Err(ExpertResidencyCatalogError::DuplicateBinding {
                    identity,
                    binding: parameter.binding_name.clone(),
                });
            }
            if !targets.insert(parameter.logical_target.as_str()) {
                return Err(ExpertResidencyCatalogError::DuplicateLogicalTarget {
                    identity,
                    target: parameter.logical_target.clone(),
                });
            }
            if let ExpertParameterRole::QuantizableProjection {
                scales_binding,
                biases_binding,
            } = parameter.role()
            {
                for companion in [scales_binding, biases_binding] {
                    if parameters
                        .iter()
                        .any(|candidate| candidate.binding_name() == companion)
                    {
                        return Err(
                            ExpertResidencyCatalogError::QuantizationCompanionCollision {
                                binding: parameter.binding_name.clone(),
                                companion: companion.clone(),
                            },
                        );
                    }
                }
            }
        }
        Ok(Self {
            identity,
            owner_group,
            owner_unit,
            unit_path,
            distribution,
            parameters,
            byte_len: None,
        })
    }

    /// Returns the global cache identity selected by architecture routing.
    pub const fn identity(&self) -> ParameterBankKey {
        self.identity
    }

    /// Returns the canonical execution group that owns this expert.
    pub const fn owner_group(&self) -> &ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the architecture-global unit index inside the owning group.
    pub const fn owner_unit(&self) -> usize {
        self.owner_unit
    }

    /// Returns the architecture execution-unit path that owns these parameters.
    pub fn unit_path(&self) -> &str {
        &self.unit_path
    }

    /// Returns how this expert participates in expert-parallel placement.
    pub const fn distribution(&self) -> ExpertResidencyDistribution {
        self.distribution
    }

    /// Returns every exact expert-local parameter recipe.
    pub fn parameters(&self) -> &[ExpertParameterRecipe] {
        &self.parameters
    }

    /// Attaches exact admitted materialized bytes when already derived upstream.
    pub fn with_byte_len(mut self, byte_len: u64) -> Result<Self, ExpertResidencyCatalogError> {
        if byte_len == 0 {
            return Err(ExpertResidencyCatalogError::InvalidByteGeometry {
                identity: self.identity,
                detail: "materialized byte count is zero".into(),
            });
        }
        self.byte_len = Some(byte_len);
        Ok(self)
    }

    /// Returns exact admitted materialized bytes for this atomic unit.
    pub const fn byte_len(&self) -> Option<u64> {
        self.byte_len
    }

    /// Consumes the unit into its parameter recipes.
    pub fn into_parameters(self) -> Vec<ExpertParameterRecipe> {
        self.parameters
    }
}

/// Complete architecture-owned schedule for independent expert residency.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertResidencyCatalog {
    units: Vec<ExpertResidencyUnit>,
}

impl ExpertResidencyCatalog {
    /// Validates a non-empty catalog with globally unique cache identities.
    pub fn new(
        units: impl IntoIterator<Item = ExpertResidencyUnit>,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let units = units.into_iter().collect::<Vec<_>>();
        if units.is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyCatalog);
        }
        let mut identities = BTreeSet::new();
        for unit in &units {
            if !identities.insert(unit.identity) {
                return Err(ExpertResidencyCatalogError::DuplicateIdentity(
                    unit.identity,
                ));
            }
        }
        Ok(Self { units })
    }

    /// Returns the deterministic architecture order of resident expert units.
    pub fn units(&self) -> &[ExpertResidencyUnit] {
        &self.units
    }

    /// Returns one atomic unit by its architecture-translated bank identity.
    pub fn unit(&self, identity: ParameterBankKey) -> Option<&ExpertResidencyUnit> {
        self.units.iter().find(|unit| unit.identity == identity)
    }

    /// Returns every canonical parameter assigned to addressable storage.
    pub fn logical_targets(&self) -> BTreeSet<&str> {
        self.units
            .iter()
            .flat_map(|unit| unit.parameters.iter())
            .map(ExpertParameterRecipe::logical_target)
            .collect()
    }

    /// Infers and retains exact materialized bytes from admitted recipe metadata.
    pub fn with_inferred_byte_geometry<C: RecipeCatalog + ?Sized>(
        mut self,
        catalog: &C,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        for unit in &mut self.units {
            let bytes = unit
                .parameters
                .iter_mut()
                .try_fold(0u64, |total, parameter| {
                    let metadata = parameter.recipe().infer(catalog).map_err(|error| {
                        ExpertResidencyCatalogError::InvalidByteGeometry {
                            identity: unit.identity,
                            detail: error.to_string(),
                        }
                    })?;
                    let bytes = metadata.byte_len;
                    parameter.metadata = Some(metadata);
                    total.checked_add(bytes).ok_or_else(|| {
                        ExpertResidencyCatalogError::InvalidByteGeometry {
                            identity: unit.identity,
                            detail: "materialized byte count overflowed".into(),
                        }
                    })
                })?;
            if bytes == 0 {
                return Err(ExpertResidencyCatalogError::InvalidByteGeometry {
                    identity: unit.identity,
                    detail: "materialized byte count is zero".into(),
                });
            }
            unit.byte_len = Some(bytes);
        }
        Ok(self)
    }

    /// Consumes the catalog into its deterministic architecture order.
    pub fn into_units(self) -> Vec<ExpertResidencyUnit> {
        self.units
    }

    /// Consumes the catalog and retains units owned by a caller's execution partition.
    ///
    /// Selection is expressed only in the architecture's canonical group-local address
    /// space so adapters do not flatten heterogeneous execution groups back into layer
    /// ordinals.
    pub fn into_units_selected_by_owner(
        self,
        mut owns_unit: impl FnMut(&ExecutionGroupId, usize) -> bool,
    ) -> impl Iterator<Item = ExpertResidencyUnit> {
        self.units
            .into_iter()
            .filter(move |unit| owns_unit(unit.owner_group(), unit.owner_unit()))
    }
}

impl IntoIterator for ExpertResidencyCatalog {
    type Item = ExpertResidencyUnit;
    type IntoIter = std::vec::IntoIter<ExpertResidencyUnit>;

    fn into_iter(self) -> Self::IntoIter {
        self.units.into_iter()
    }
}

/// Invalid architecture-owned expert residency topology.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExpertResidencyCatalogError {
    /// A local acquired-bank name is empty.
    #[error("expert residency binding name must not be empty")]
    EmptyBindingName,
    /// A binding has no exact architecture destination.
    #[error("expert residency binding {binding:?} has an empty logical target")]
    EmptyLogicalTarget {
        /// Invalid local binding name.
        binding: String,
    },
    /// A binding recipe has no checkpoint inputs.
    #[error("expert residency binding {binding:?} has no checkpoint recipe source")]
    EmptyRecipe {
        /// Invalid local binding name.
        binding: String,
    },
    /// A quantizable binding did not declare a usable companion name.
    #[error("expert residency binding {binding:?} has an empty quantization companion")]
    EmptyQuantizationCompanion {
        /// Invalid projection binding.
        binding: String,
    },
    /// A packed companion collides with its projection or another declared binding.
    #[error("expert residency binding {binding:?} quantization companion {companion:?} collides with an existing binding")]
    QuantizationCompanionCollision {
        /// Quantizable projection binding.
        binding: String,
        /// Colliding packed companion binding.
        companion: String,
    },
    /// An expert is not attached to an architecture execution unit.
    #[error("expert {identity:?} has an empty architecture unit path")]
    EmptyUnitPath {
        /// Invalid expert identity.
        identity: ParameterBankKey,
    },
    /// An expert has no checkpoint-backed parameters.
    #[error("expert {identity:?} has no residency parameters")]
    EmptyUnit {
        /// Invalid expert identity.
        identity: ParameterBankKey,
    },
    /// Admitted recipes did not yield a finite nonzero atomic byte count.
    #[error("expert {identity:?} has invalid materialized byte geometry: {detail}")]
    InvalidByteGeometry {
        /// Invalid unit.
        identity: ParameterBankKey,
        /// Metadata inference failure.
        detail: String,
    },
    /// One acquired bank name is repeated inside an expert.
    #[error("expert {identity:?} repeats local binding {binding:?}")]
    DuplicateBinding {
        /// Invalid expert identity.
        identity: ParameterBankKey,
        /// Repeated local binding.
        binding: String,
    },
    /// One architecture target is repeated inside an expert.
    #[error("expert {identity:?} repeats logical target {target:?}")]
    DuplicateLogicalTarget {
        /// Invalid expert identity.
        identity: ParameterBankKey,
        /// Repeated logical destination.
        target: String,
    },
    /// No independently resident experts were declared.
    #[error("architecture declares no independently resident experts")]
    EmptyCatalog,
    /// Two units use the same router/cache identity.
    #[error("architecture repeats expert residency identity {0:?}")]
    DuplicateIdentity(ParameterBankKey),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ParallelTopology;

    #[test]
    fn expert_realization_is_the_complete_balanced_owner_map() {
        let topology = ParallelTopology::new(1, 1, 3, 1).unwrap();
        let plans = (0..3)
            .map(|rank| {
                let decoder = ExecutionGroupId::new("text_decoder").unwrap();
                ExpertRealizationPlan::balanced(
                    8,
                    ParallelRankTopology::new(topology, rank).unwrap(),
                    BTreeMap::from([((decoder, 4), format!("rank-{rank}-bank"))]),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(plans[0].owners(), [0, 0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(plans[0].local_global_group_indices(), [0, 1, 2]);
        assert_eq!(plans[1].local_global_group_indices(), [3, 4, 5]);
        assert_eq!(plans[2].local_global_group_indices(), [6, 7]);
        assert_eq!(
            plans[2].unit_spec("text_decoder", 4).map(String::as_str),
            Some("rank-2-bank")
        );
        assert!(plans[2].has_routed_units_in_group("text_decoder"));
        assert!(!plans[2].has_routed_units_in_group("mtp.0"));
    }

    #[test]
    fn expert_realization_rejects_empty_ranks_and_unit_schedules() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 3, 1).unwrap(), 0).unwrap();
        assert!(matches!(
            ExpertRealizationPlan::<()>::balanced(
                2,
                rank,
                BTreeMap::from([((ExecutionGroupId::new("decoder").unwrap(), 0), ())])
            ),
            Err(ExpertRealizationPlanError::InvalidTopology(_))
        ));
        assert!(matches!(
            ExpertRealizationPlan::<()>::balanced(3, rank, BTreeMap::new()),
            Err(ExpertRealizationPlanError::EmptyUnitSchedule)
        ));
    }
}
