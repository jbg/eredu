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
    role: ExpertParameterRole,
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

    /// Consumes the declaration into its local name, logical target, recipe, and role.
    pub fn into_parts(self) -> (String, String, DerivedWeightRecipe, ExpertParameterRole) {
        (
            self.binding_name,
            self.logical_target,
            self.recipe,
            self.role,
        )
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
