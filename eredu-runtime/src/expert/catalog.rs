//! Validated independently leased bank entry declarations.
use crate::{OffloadUnit, ParameterBankKey};
use eredu_core::residency::OffloadUnitId;

/// Invalid independent parameter storage identity or admitted byte extent.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AddressableBankEntryError {
    /// A stored member must retain a nonzero admitted payload extent.
    #[error("entry {key:?} must contain at least one byte")]
    ZeroBytes {
        /// Invalid global member identity.
        key: ParameterBankKey,
    },
    /// The residency unit belongs to another global storage identity.
    #[error("entry {key:?} requires unit id {expected}, got {actual}")]
    UnitIdentityMismatch {
        /// Declared global member identity.
        key: ParameterBankKey,
        /// Canonical independent residency unit.
        expected: OffloadUnitId,
        /// Supplied residency unit.
        actual: OffloadUnitId,
    },
}

/// One independently leased storage member and its admitted atomic payload bound.
/// Native tensors and completion resources belong to the selected backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressableBankEntry {
    key: ParameterBankKey,
    unit: OffloadUnit,
    bytes: u64,
}

impl AddressableBankEntry {
    /// Validates nonzero payload and exact key-to-unit identity. Executable
    /// geometry, combined budgets and duplicate catalog ownership are admitted
    /// separately against the selected bindings and residency plan.
    pub fn new(
        key: ParameterBankKey,
        unit: OffloadUnit,
        bytes: u64,
    ) -> Result<Self, AddressableBankEntryError> {
        if bytes == 0 {
            return Err(AddressableBankEntryError::ZeroBytes { key });
        }
        let expected = key.unit_id();
        if unit.id() != &expected {
            return Err(AddressableBankEntryError::UnitIdentityMismatch {
                key,
                expected,
                actual: unit.id().clone(),
            });
        }
        Ok(Self { key, unit, bytes })
    }
    /// Original global storage member identity.
    pub const fn key(&self) -> ParameterBankKey {
        self.key
    }
    /// Exact immutable independently leased binding declaration.
    pub const fn unit(&self) -> &OffloadUnit {
        &self.unit
    }
    /// Admitted atomic payload byte extent at this materialization stage.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
    /// Consumes the complete retained storage declaration.
    pub fn into_parts(self) -> (ParameterBankKey, OffloadUnit, u64) {
        (self.key, self.unit, self.bytes)
    }
}

/// Failure assembling selected independent storage from canonical binding plans.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankCatalogError {
    /// Multiple plans claim the same independently leased physical member.
    #[error("selected bank catalog repeats member {key:?}")]
    DuplicateMember {
        /// Repeated global member identity.
        key: ParameterBankKey,
    },
    /// A binding plan cannot form a valid residency unit.
    #[error(transparent)]
    Declaration(#[from] crate::ResidencyDeclarationError),
    /// A selected independent entry declaration is invalid.
    #[error(transparent)]
    Entry(#[from] AddressableBankEntryError),
}

/// Neutral selected bank declarations for construction and parameter operations.
/// Source, native capability, namespace and residency-budget admission consume
/// these declarations before execution; the catalog itself owns no native value.
#[derive(Debug, Clone)]
pub struct AddressableBankCatalog {
    /// Canonical destinations for local bindings and generated companions.
    pub parameter_targets: std::collections::BTreeMap<(ParameterBankKey, String), String>,
    /// Independent source-backed entries before selected load-time conversion.
    pub entries: Vec<AddressableBankEntry>,
    /// Exact selected encoding and precision for each transforming binding.
    pub transformations:
        std::collections::BTreeMap<(ParameterBankKey, String), super::AddressableBindingTransform>,
    /// Selected executable byte extent of each independent member.
    pub expected_bytes: std::collections::BTreeMap<ParameterBankKey, u64>,
    /// Selected ownership, distribution and actual rank of each member.
    pub placements:
        std::collections::BTreeMap<ParameterBankKey, super::AddressableBankMemberPlacement>,
}

impl AddressableBankCatalog {
    /// Consumes exact member plans, rejecting duplicate physical ownership and
    /// retaining entry order, canonical destinations, transforms and byte geometry.
    pub fn from_binding_plans(
        plans: impl IntoIterator<Item = super::AddressableBankBindingPlan>,
    ) -> Result<Self, AddressableBankCatalogError> {
        let plans = plans.into_iter();
        let mut catalog = Self {
            parameter_targets: Default::default(),
            entries: Vec::with_capacity(plans.size_hint().0),
            transformations: Default::default(),
            expected_bytes: Default::default(),
            placements: Default::default(),
        };
        for plan in plans {
            let (key, bindings, transforms, footprint, placement, targets) = plan.into_parts();
            if catalog.expected_bytes.contains_key(&key) {
                return Err(AddressableBankCatalogError::DuplicateMember { key });
            }
            catalog.parameter_targets.extend(
                targets
                    .into_iter()
                    .map(|(local, target)| ((key, local), target)),
            );
            catalog.transformations.extend(
                transforms
                    .into_iter()
                    .map(|(binding, transform)| ((key, binding), transform)),
            );
            catalog.entries.push(AddressableBankEntry::new(
                key,
                OffloadUnit::new(key.unit_id(), bindings)?,
                footprint.source_bytes(),
            )?);
            catalog
                .expected_bytes
                .insert(key, footprint.selected_bytes());
            catalog.placements.insert(key, placement);
        }
        Ok(catalog)
    }
}

/// Failure admitting independent members and compact rows to one residency pool.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankResidencyError {
    /// A physical member occurs more than once in the pool.
    #[error("duplicate bank residency member {key:?}")]
    DuplicateMember {
        /// Repeated global storage identity.
        key: ParameterBankKey,
    },
    /// The pool owns neither an independent member nor a row namespace.
    #[error("bank residency requires at least one member or row namespace")]
    EmptyCatalog,
    /// Member geometry, namespace overlap or configured budgets are invalid.
    #[error(transparent)]
    Plan(#[from] eredu_core::residency::OffloadError),
}

/// Admits exact member extents and compact row namespaces without reading payloads.
/// Selected executable extents can be admitted before source transformations;
/// materialized extents are admitted again when preparing native cache inputs.
pub fn plan_addressable_bank_residency(
    entries: impl IntoIterator<Item = (ParameterBankKey, u64)>,
    rows: &[crate::RowResidencyRange],
    config: eredu_core::residency::OffloadConfig,
    policy: eredu_core::residency::ResidencyPolicy,
    initial_tier: eredu_core::residency::MemoryTier,
) -> Result<eredu_core::residency::OffloadPlan, AddressableBankResidencyError> {
    use eredu_core::residency::{OffloadPlan, OffloadUnitSpec};
    let mut keys = std::collections::BTreeSet::new();
    let mut specs = Vec::new();
    for (key, bytes) in entries {
        if !keys.insert(key) {
            return Err(AddressableBankResidencyError::DuplicateMember { key });
        }
        specs.push(OffloadUnitSpec::new(
            key.unit_id(),
            bytes,
            policy,
            initial_tier,
        )?);
    }
    if keys.is_empty() && rows.is_empty() {
        return Err(AddressableBankResidencyError::EmptyCatalog);
    }
    Ok(OffloadPlan::with_ranges(
        config,
        specs,
        rows.iter().map(|row| row.range().clone()),
    )?)
}

/// Consuming, backend-neutral cache inputs with admitted ownership and budgets.
/// Each member remains an independent lease; compact rows retain their exact sources.
pub struct PreparedAddressableBankResidency {
    catalog: std::collections::BTreeMap<ParameterBankKey, u64>,
    definitions: Vec<OffloadUnit>,
    rows: Vec<crate::RowResidencyRange>,
    plan: eredu_core::residency::OffloadPlan,
}

impl PreparedAddressableBankResidency {
    /// Consumes final binding declarations and validates the combined residency plan.
    pub fn new(
        entries: impl IntoIterator<Item = AddressableBankEntry>,
        rows: Vec<crate::RowResidencyRange>,
        config: eredu_core::residency::OffloadConfig,
        policy: eredu_core::residency::ResidencyPolicy,
        initial_tier: eredu_core::residency::MemoryTier,
    ) -> Result<Self, AddressableBankResidencyError> {
        let entries = entries.into_iter();
        let mut catalog = std::collections::BTreeMap::new();
        let mut definitions = Vec::with_capacity(entries.size_hint().0);
        let plan = plan_addressable_bank_residency(
            entries.map(|entry| {
                let (key, unit, bytes) = entry.into_parts();
                catalog.insert(key, bytes);
                definitions.push(unit);
                (key, bytes)
            }),
            &rows,
            config,
            policy,
            initial_tier,
        )?;
        Ok(Self {
            catalog,
            definitions,
            rows,
            plan,
        })
    }

    /// Moves admitted inputs into the backend's existing residency manager.
    pub fn into_parts(
        self,
    ) -> (
        std::collections::BTreeMap<ParameterBankKey, u64>,
        Vec<OffloadUnit>,
        Vec<crate::RowResidencyRange>,
        eredu_core::residency::OffloadPlan,
    ) {
        (self.catalog, self.definitions, self.rows, self.plan)
    }
}
