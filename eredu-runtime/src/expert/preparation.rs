//! Consuming selected bank admission and post-materialization preparation.
use super::{
    AddressableBankCatalog, AddressableBankEntry, AddressableBankMemberPlacement,
    AddressableBankResidencyError, AddressableBankSelectionError, AddressableBankTransformPlan,
    PreparedAddressableBankResidency, plan_addressable_bank_residency,
    validate_addressable_bank_selection,
};
use crate::{ParameterBankKey, RowResidencyRange};
use crate::parameter_operations::{
    BankParameterPreparationError, BankParameterIndexError, PreparedBankParameterIndex,
    prepare_bank_parameter_members,
};
use eredu_checkpoint::recipe::RecipeCatalog;
use eredu_core::residency::{MemoryTier, OffloadConfig, ResidencyPolicy};
use std::collections::{BTreeMap, BTreeSet};

/// Invalid selected or returned independent bank resources.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankPreparationError {
    /// Cold selected ownership or geometry is inconsistent.
    #[error(transparent)]
    Selection(#[from] AddressableBankSelectionError),
    /// Independent storage or row residency admission failed.
    #[error(transparent)]
    Residency(#[from] AddressableBankResidencyError),
    /// Final bindings cannot be joined to their canonical parameter identities.
    #[error(transparent)]
    Parameter(#[from] BankParameterPreparationError),
    /// Final physical bindings have ambiguous canonical ownership.
    #[error(transparent)]
    Index(#[from] BankParameterIndexError),
    /// The materializer returned a different set of physical members.
    #[error("materialized bank omits {missing:?} and unexpectedly returns {unexpected:?}")]
    MemberCoverage {
        /// Selected members absent from the result.
        missing: BTreeSet<ParameterBankKey>,
        /// Returned members outside the retained selection.
        unexpected: BTreeSet<ParameterBankKey>,
    },
    /// A returned member differs from its selected executable extent.
    #[error("materialized bank member {key:?} bytes differ: expected {expected}, got {actual}")]
    ByteMismatch {
        /// Original physical member identity.
        key: ParameterBankKey,
        /// Selected executable extent.
        expected: u64,
        /// Returned admitted extent.
        actual: u64,
    },
}

/// Retained selection authority across backend materialization. Consuming binding
/// finalizes one retained selection exactly once.
/// Native operator and checkpoint-format support are admitted by the backend first.
pub struct AddressableBankPreparation {
    targets: BTreeMap<(ParameterBankKey, String), String>,
    placements: BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
    expected: BTreeMap<ParameterBankKey, u64>,
    rows: Vec<RowResidencyRange>,
    config: OffloadConfig,
}

impl AddressableBankPreparation {
    /// Admits selected executable geometry and budgets without reading payloads,
    /// returning exact source entries and the retained ordinary conversion plan.
    pub fn admit<C: RecipeCatalog + ?Sized>(
        selected: AddressableBankCatalog,
        rows: Vec<RowResidencyRange>,
        config: OffloadConfig,
        source: &C,
    ) -> Result<
        (
            Vec<AddressableBankEntry>,
            AddressableBankTransformPlan,
            Self,
        ),
        AddressableBankPreparationError,
    > {
        validate_addressable_bank_selection(
            selected
                .entries
                .iter()
                .map(|entry| (entry.key(), entry.unit().bindings())),
            &selected.placements,
            &selected.expected_bytes,
            &selected.transformations,
            source,
        )?;
        let mut expected = selected.expected_bytes;
        expected.retain(|key, _| selected.placements.contains_key(key));
        plan_addressable_bank_residency(
            selected
                .entries
                .iter()
                .map(|entry| (entry.key(), expected[&entry.key()])),
            &rows,
            config,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )?;
        Ok((
            selected.entries,
            AddressableBankTransformPlan::new(selected.transformations),
            Self {
                targets: selected.parameter_targets,
                placements: selected.placements,
                expected,
                rows,
                config,
            },
        ))
    }

    /// Consumes final materializer bindings against the retained selection. Direct
    /// and converted resources use the same identity, byte and metadata admission.
    pub fn bind<C: RecipeCatalog + ?Sized>(
        self,
        entries: Vec<AddressableBankEntry>,
        source: &C,
    ) -> Result<PreparedSelectedAddressableBank, AddressableBankPreparationError> {
        let expected_keys: BTreeSet<_> = self.expected.keys().copied().collect();
        let actual_keys: BTreeSet<_> = entries.iter().map(AddressableBankEntry::key).collect();
        if expected_keys != actual_keys {
            return Err(AddressableBankPreparationError::MemberCoverage {
                missing: expected_keys.difference(&actual_keys).copied().collect(),
                unexpected: actual_keys.difference(&expected_keys).copied().collect(),
            });
        }
        for entry in &entries {
            let expected = self.expected[&entry.key()];
            if entry.bytes() != expected {
                return Err(AddressableBankPreparationError::ByteMismatch {
                    key: entry.key(),
                    expected,
                    actual: entry.bytes(),
                });
            }
        }
        let members = prepare_bank_parameter_members(
            entries
                .iter()
                .map(|entry| (entry.key(), entry.unit().bindings())),
            &self.targets,
            source,
        )?;
        let members = PreparedBankParameterIndex::new(members)?;
        let residency = PreparedAddressableBankResidency::new(
            entries,
            self.rows,
            self.config,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )?;
        Ok(PreparedSelectedAddressableBank {
            residency,
            members,
            placements: self.placements,
        })
    }
}

/// Final neutral resources shared by direct and converted selected bank execution.
pub struct PreparedSelectedAddressableBank {
    residency: PreparedAddressableBankResidency,
    members: PreparedBankParameterIndex,
    placements: BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
}
impl PreparedSelectedAddressableBank {
    /// Moves exact cache inputs, canonical parameter metadata and ownership to the backend.
    pub fn into_parts(
        self,
    ) -> (
        PreparedAddressableBankResidency,
        PreparedBankParameterIndex,
        BTreeMap<ParameterBankKey, AddressableBankMemberPlacement>,
    ) {
        (self.residency, self.members, self.placements)
    }
}
