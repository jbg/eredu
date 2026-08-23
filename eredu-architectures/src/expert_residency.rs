//! Backend-neutral topology and checkpoint recipes for independent expert residency.

use std::collections::BTreeSet;

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_runtime::{ExecutionGroupId, ExpertIdentity};

/// Placement of one expert relative to an expert-parallel axis.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertResidencyDistribution {
    /// Assign the expert by its global router identity across expert ranks.
    ExpertParallel,
    /// Materialize the expert on every rank that owns its execution unit.
    Replicated,
}

/// One architecture-logical parameter in an independently resident expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertParameterRecipe {
    binding_name: String,
    logical_target: String,
    recipe: DerivedWeightRecipe,
}

impl ExpertParameterRecipe {
    /// Creates one exact local binding and its architecture-logical destination.
    pub fn new(
        binding_name: impl Into<String>,
        logical_target: impl Into<String>,
        recipe: DerivedWeightRecipe,
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
        Ok(Self {
            binding_name,
            logical_target,
            recipe,
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

    /// Consumes the declaration into its local name, logical target, and recipe.
    pub fn into_parts(self) -> (String, String, DerivedWeightRecipe) {
        (self.binding_name, self.logical_target, self.recipe)
    }
}

/// One independently addressable expert and its owning execution unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertResidencyUnit {
    identity: ExpertIdentity,
    owner_group: ExecutionGroupId,
    owner_unit: usize,
    unit_path: String,
    distribution: ExpertResidencyDistribution,
    parameters: Vec<ExpertParameterRecipe>,
}

impl ExpertResidencyUnit {
    /// Creates one complete atomic expert unit.
    pub fn new(
        identity: ExpertIdentity,
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
        }
        Ok(Self {
            identity,
            owner_group,
            owner_unit,
            unit_path,
            distribution,
            parameters,
        })
    }

    /// Returns the global cache identity selected by architecture routing.
    pub const fn identity(&self) -> ExpertIdentity {
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

    /// Consumes the catalog into its deterministic architecture order.
    pub fn into_units(self) -> Vec<ExpertResidencyUnit> {
        self.units
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
    /// An expert is not attached to an architecture execution unit.
    #[error("expert {identity:?} has an empty architecture unit path")]
    EmptyUnitPath {
        /// Invalid expert identity.
        identity: ExpertIdentity,
    },
    /// An expert has no checkpoint-backed parameters.
    #[error("expert {identity:?} has no residency parameters")]
    EmptyUnit {
        /// Invalid expert identity.
        identity: ExpertIdentity,
    },
    /// One acquired bank name is repeated inside an expert.
    #[error("expert {identity:?} repeats local binding {binding:?}")]
    DuplicateBinding {
        /// Invalid expert identity.
        identity: ExpertIdentity,
        /// Repeated local binding.
        binding: String,
    },
    /// One architecture target is repeated inside an expert.
    #[error("expert {identity:?} repeats logical target {target:?}")]
    DuplicateLogicalTarget {
        /// Invalid expert identity.
        identity: ExpertIdentity,
        /// Repeated logical destination.
        target: String,
    },
    /// No independently resident experts were declared.
    #[error("architecture declares no independently resident experts")]
    EmptyCatalog,
    /// Two units use the same router/cache identity.
    #[error("architecture repeats expert residency identity {0:?}")]
    DuplicateIdentity(ExpertIdentity),
}
