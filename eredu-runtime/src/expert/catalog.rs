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
