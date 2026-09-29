//! Payload-lazy selection of exact row mechanisms and shared workspace budgets.
use super::*;
use crate::{AddressableStorageCapabilities, ParameterBankLoadOptions};
use std::collections::BTreeMap;

/// Native temporary storage for one exact prepared row contract. These are
/// conservative peaks, excluding residency, retained scalars and portable output
/// planning, which the selection driver accounts separately.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowLookupWorkspace {
    /// Decode/copy workspace for the largest permitted acquisition.
    pub decode_bytes: u64,
    /// Workspace for materializing the exact scalar recipe during binding.
    pub scalar_bytes: u64,
}

/// Side-effect-free support for a single exact row contract. Implementations must
/// inspect retained metadata only: no payload reads, native tensors or devices.
pub trait RowLookupMechanismSupport {
    /// Storage tiers, access classes and completion ownership.
    fn storage(&self) -> Option<AddressableStorageCapabilities>;
    /// Buffer and retention facts for one maximum-size acquisition. The default
    /// preserves logical geometry with explicitly unknown physical storage.
    fn decode_memory(
        &self,
        prepared: &RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, RowLookupError> {
        prepared.decode_memory()
    }
    /// Returns no mechanism for an unsupported encoding, arithmetic boundary or
    /// companion recipe. Workspace includes every native conversion intermediate.
    fn workspace(
        &self,
        prepared: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError>;
}

/// Selected resource facts. Lookups execute serially; their workspace is a peak,
/// while retained scalar allocations are cumulative across all table bindings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SelectedRowLookupRequirements {
    /// Source, companion, acquisition and portable invocation allowances.
    pub rows: RowLookupRequirements,
    /// Maximum native decode workspace of one lookup.
    pub decode_bytes: u64,
    /// Maximum workspace for one scalar's preparation.
    pub scalar_preparation_bytes: u64,
    /// Maximum simultaneous portable planning, output and native decode storage.
    pub invocation_bytes: u64,
}

/// A typed cold failure, before any scalar or table payload is materialized.
#[derive(Debug, thiserror::Error)]
pub enum RowLookupSelectionError {
    /// An exact storage or decoding mechanism is missing.
    #[error("row lookup mechanism {mechanism} is unavailable for {parameter:?}")]
    Mechanism {
        /// Exact table, or none for a collection-wide facility.
        parameter: Option<ParameterId>,
        /// Missing facility.
        mechanism: &'static str,
    },
    /// Binding attempted to replace the selected pool configuration.
    #[error("row lookup pool configuration differs from cold selection")]
    PoolMismatch,
    /// Bound sources do not match the admitted metadata contract.
    #[error("prepared rows differ from selected header descriptors")]
    DescriptorMismatch,
    /// Geometry, workspace or retained source validation failed.
    #[error(transparent)]
    Lookup(#[from] RowLookupError),
    /// Generic parameter-bank controls are invalid.
    #[error(transparent)]
    Policy(#[from] crate::WeightResidencyPolicyError),
}

/// Metadata-only selected mechanisms and bounds. Binding readable sources checks
/// exact descriptors and never repeats mechanism selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedRowLookupPlans {
    descriptors: RowLookupDescriptors,
    options: ParameterBankLoadOptions,
    storage: Option<AddressableStorageCapabilities>,
    workspaces: BTreeMap<ParameterId, RowLookupWorkspace>,
    decode_memory: BTreeMap<ParameterId, eredu_nn::mechanism_memory::MechanismMemoryContract>,
    requirements: SelectedRowLookupRequirements,
}
impl SelectedRowLookupPlans {
    /// Selects bounded row execution using generic parameter-bank controls. The
    /// compact-bank scratch limit covers portable planning/output AND native
    /// decode storage together, rather than granting each a separate full limit.
    pub fn select(
        descriptors: RowLookupDescriptors,
        options: ParameterBankLoadOptions,
        scalar_limit: u64,
        support: &impl RowLookupMechanismSupport,
    ) -> Result<Self, RowLookupSelectionError> {
        options.validate()?;
        descriptors.validate_pool_budget(options.offload())?;
        let mut result = Self {
            requirements: SelectedRowLookupRequirements {
                rows: descriptors.requirements(),
                ..Default::default()
            },
            descriptors,
            options,
            storage: None,
            workspaces: BTreeMap::new(),
            decode_memory: BTreeMap::new(),
        };
        if result.descriptors.entries().is_empty() {
            return Ok(result);
        }
        let missing = |mechanism| RowLookupSelectionError::Mechanism {
            parameter: None,
            mechanism,
        };
        let storage = support
            .storage()
            .ok_or_else(|| missing("addressable storage"))?;
        result.storage = Some(storage);
        for (available, mechanism) in [
            (storage.bulk_access(), "bulk row access"),
            (storage.incremental_access(), "incremental row access"),
            (storage.lease_completion(), "row lease completion"),
            (storage.tiers().device(), "device row storage"),
            (storage.tiers().disk(), "checkpoint row storage"),
            (
                options.offload().host_budget_bytes() == Some(0) || storage.tiers().host(),
                "host row storage",
            ),
        ] {
            if !available {
                return Err(missing(mechanism));
            }
        }
        admit(
            "retained row scalars",
            result.requirements.rows.scalar_bytes,
            scalar_limit,
        )?;
        for (id, entry) in result.descriptors.entries() {
            let workspace =
                support
                    .workspace(entry)?
                    .ok_or_else(|| RowLookupSelectionError::Mechanism {
                        parameter: Some(id.clone()),
                        mechanism: "exact row decoder and companion recipe",
                    })?;
            let memory = support.decode_memory(entry)?;
            memory.validate().map_err(RowLookupError::from)?;
            if memory.values != entry.decode_memory()?.values {
                return Err(RowLookupSelectionError::Mechanism {
                    parameter: Some(id.clone()),
                    mechanism: "row decoder memory geometry differs from retained source",
                });
            }
            result.decode_memory.insert(id.clone(), memory);
            // Empty workspace cannot describe a materialized, nonempty result.
            if workspace.decode_bytes == 0
                || (entry.scale().is_some() && workspace.scalar_bytes == 0)
            {
                return Err(RowLookupError::Geometry.into());
            }
            let invocation = entry
                .limits()
                .host_bytes
                .checked_add(entry.limits().output_bytes)
                .and_then(|n| n.checked_add(workspace.decode_bytes))
                .ok_or(RowLookupError::Geometry)?;
            let required = invocation.max(workspace.scalar_bytes);
            admit(
                "row workspace",
                required,
                options.compact_bank_scratch_bytes(),
            )?;
            admit(
                "backend row workspace",
                required,
                storage.maximum_compact_bytes(),
            )?;
            result.requirements.decode_bytes =
                result.requirements.decode_bytes.max(workspace.decode_bytes);
            result.requirements.scalar_preparation_bytes = result
                .requirements
                .scalar_preparation_bytes
                .max(workspace.scalar_bytes);
            result.requirements.invocation_bytes =
                result.requirements.invocation_bytes.max(invocation);
            result.workspaces.insert(id.clone(), workspace);
        }
        Ok(result)
    }
    /// Selected cold descriptors, with no readable sources.
    pub fn descriptors(&self) -> &RowLookupDescriptors {
        &self.descriptors
    }
    /// Cold resource totals selected before source binding.
    pub const fn requirements(&self) -> SelectedRowLookupRequirements {
        self.requirements
    }
    /// Shared parameter-bank pool and scratch policy selected before binding.
    pub const fn options(&self) -> ParameterBankLoadOptions {
        self.options
    }
    /// Exact decoder workspace selected for one parameter.
    pub fn workspace(&self, parameter: &ParameterId) -> Option<RowLookupWorkspace> {
        self.workspaces.get(parameter).copied()
    }
    /// Native decoder memory facts retained by cold admission.
    pub fn decode_memory(
        &self,
    ) -> &BTreeMap<ParameterId, eredu_nn::mechanism_memory::MechanismMemoryContract> {
        &self.decode_memory
    }
    /// Retains selected mechanisms for an exact global execution-unit range.
    /// Unit identities remain global; filtering never reopens sources or repeats
    /// backend mechanism selection.
    pub fn for_units(
        mut self,
        units: std::ops::Range<usize>,
        execution_units: usize,
    ) -> Result<Self, RowLookupSelectionError> {
        if units.start > units.end || units.end > execution_units {
            return Err(RowLookupError::Geometry.into());
        }
        self.descriptors = RowLookupDescriptors::new(
            self.descriptors
                .entries()
                .values()
                .filter(|entry| units.contains(&entry.spec().unit))
                .cloned(),
            execution_units,
        )?;
        self.workspaces
            .retain(|id, _| self.descriptors.entries().contains_key(id));
        self.decode_memory
            .retain(|id, _| self.descriptors.entries().contains_key(id));
        self.requirements = SelectedRowLookupRequirements {
            rows: self.descriptors.requirements(),
            ..Default::default()
        };
        for (id, entry) in self.descriptors.entries() {
            let workspace = self.workspaces[id];
            self.requirements.decode_bytes =
                self.requirements.decode_bytes.max(workspace.decode_bytes);
            self.requirements.scalar_preparation_bytes = self
                .requirements
                .scalar_preparation_bytes
                .max(workspace.scalar_bytes);
            let invocation = entry
                .limits()
                .host_bytes
                .checked_add(entry.limits().output_bytes)
                .and_then(|bytes| bytes.checked_add(workspace.decode_bytes))
                .ok_or(RowLookupError::Geometry)?;
            self.requirements.invocation_bytes = self.requirements.invocation_bytes.max(invocation);
        }
        Ok(self)
    }

    /// Consumes the selected header contract into exact source authority, without
    /// calling a backend or selecting mechanisms again.
    pub fn bind(
        self,
        prepared: PreparedRowLookups,
    ) -> Result<SelectedRowLookups, RowLookupSelectionError> {
        if prepared.descriptors() != &self.descriptors {
            return Err(RowLookupSelectionError::DescriptorMismatch);
        }
        Ok(SelectedRowLookups {
            prepared,
            plan: self,
        })
    }
}

/// Bound row authority retaining exact sources and its original cold selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedRowLookups {
    prepared: PreparedRowLookups,
    plan: SelectedRowLookupPlans,
}
impl SelectedRowLookups {
    /// Selects from metadata then binds the supplied exact sources through the
    /// same admission algorithm used before readable source construction.
    pub fn select(
        prepared: PreparedRowLookups,
        options: ParameterBankLoadOptions,
        scalar_limit: u64,
        support: &impl RowLookupMechanismSupport,
    ) -> Result<Self, RowLookupSelectionError> {
        SelectedRowLookupPlans::select(
            prepared.descriptors().clone(),
            options,
            scalar_limit,
            support,
        )?
        .bind(prepared)
    }
    /// Metadata-only authority retained before exact sources were bound.
    pub fn plan(&self) -> &SelectedRowLookupPlans {
        &self.plan
    }
    /// Exact prepared sources. Cloning retains their identity and provenance.
    pub fn prepared(&self) -> &PreparedRowLookups {
        &self.prepared
    }
    /// One shared pool and workspace policy.
    pub const fn options(&self) -> ParameterBankLoadOptions {
        self.plan.options
    }
    /// Cold totals to combine with ordinary weights, state and transport reports.
    pub const fn requirements(&self) -> SelectedRowLookupRequirements {
        self.plan.requirements
    }
    /// Exact native workspace selected for a table.
    pub fn workspace(&self, parameter: &ParameterId) -> Option<RowLookupWorkspace> {
        self.plan.workspaces.get(parameter).copied()
    }
    /// Retained per-decoder maximum-acquisition geometry and native buffer facts.
    /// These are independent invocation contracts, not a simultaneous memory peak.
    pub fn decode_memory(
        &self,
    ) -> &BTreeMap<ParameterId, eredu_nn::mechanism_memory::MechanismMemoryContract> {
        &self.plan.decode_memory
    }
    /// Binds one selected decoder's facts to physical pools and invocation identity.
    /// Native binding cannot replace these facts with a new description.
    pub fn describe_decode_resources(
        &self,
        parameter: &ParameterId,
        query: &crate::MechanismResourceQuery,
    ) -> Result<crate::MechanismResourceDescription, eredu_core::resources::ResourceDescriptionError>
    {
        let contract = self.plan.decode_memory.get(parameter).ok_or_else(|| {
            eredu_core::resources::ResourceDescriptionError::Invalid(
                "unselected row decoder".into(),
            )
        })?;
        crate::describe_mechanism_resources(contract.clone(), query)
    }
    /// Rejects binding through a different pool or backend mechanism. This query
    /// remains payload-lazy and cannot replace the retained sources or bounds.
    pub fn validate_binding(
        &self,
        pool: eredu_core::residency::OffloadConfig,
        support: &impl RowLookupMechanismSupport,
    ) -> Result<(), RowLookupSelectionError> {
        if pool != self.plan.options.offload() {
            return Err(RowLookupSelectionError::PoolMismatch);
        }
        if self.plan.descriptors.entries().is_empty() {
            return Ok(());
        }
        if support.storage() != self.plan.storage {
            return Err(RowLookupSelectionError::Mechanism {
                parameter: None,
                mechanism: "selected row storage facilities",
            });
        }
        for (id, descriptor) in self.plan.descriptors.entries() {
            if support.workspace(descriptor)? != self.plan.workspaces.get(id).copied() {
                return Err(RowLookupSelectionError::Mechanism {
                    parameter: Some(id.clone()),
                    mechanism: "selected native row workspace",
                });
            }
            if support.decode_memory(descriptor)? != self.plan.decode_memory[id] {
                return Err(RowLookupSelectionError::Mechanism {
                    parameter: Some(id.clone()),
                    mechanism: "selected row decoder storage or retention",
                });
            }
        }
        Ok(())
    }
}
