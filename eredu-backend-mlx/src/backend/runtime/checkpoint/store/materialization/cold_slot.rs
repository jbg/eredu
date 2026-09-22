//! One cold materialization slot admitted before its fixed destinations exist.
use super::{MaterializationPayloadShape, PreparedWeightMaterialization};
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use std::{cell::RefCell, collections::TryReserveError, fmt};

struct Initializer(MaterializationPayloadShape);
struct Slot(RefCell<Option<PreparedWeightMaterialization>>);

#[derive(thiserror::Error)]
#[error("cold materialization destinations: {cause}")]
struct ConstructionFailure {
    _prefix: PreparedWeightMaterialization,
    #[source]
    cause: TryReserveError,
}
impl fmt::Debug for ConstructionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConstructionFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}

impl SharedNativeInitializer for Initializer {
    type Output = Slot;
    type Error = ConstructionFailure;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // Only the actual node/payload is constructed, not a bank transport.
        let bytes =
            PreparedWeightMaterialization::bank_layout_with_payload::<Self, Self::Error>(1, self.0)
                .ok_or(WorkingMemoryError::Overflow)?
                .prepared_slot_control_bytes;
        usize::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Slot, ConstructionFailure> {
        PreparedWeightMaterialization::try_new_source(custody, self.0)
            .map(|ready| Slot(RefCell::new(Some(ready))))
            .map_err(|(prefix, cause)| ConstructionFailure {
                _prefix: prefix,
                cause,
            })
    }
}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub(crate) struct ColdMaterializationSlotError(SharedNativeInitializationError<Initializer>);

impl ColdMaterializationSlotError {
    pub(crate) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        eredu_core::BackendFailure::from_error(self.0.into_parts().1.retire_output_and_map_error(
            |error| {
                let ConstructionFailure { _prefix, cause } = error;
                drop(_prefix);
                cause
            },
        ))
    }
}

/// A move-only source-funded slot. Its one checkout transfers the actual
/// prepared node; the node keeps its account independently of this wrapper.
pub(crate) struct ColdMaterializationSlot(InitializedSharedNative<Slot>);
impl fmt::Debug for ColdMaterializationSlot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ColdMaterializationSlot")
            .field("original_bytes", &self.0.original_bytes())
            .finish_non_exhaustive()
    }
}
impl ColdMaterializationSlot {
    pub(crate) fn required_bytes(
        shape: MaterializationPayloadShape,
    ) -> Result<u64, WorkingMemoryError> {
        MemoryLedger::shared_native_initialization_required_bytes(&Initializer(shape))
    }

    pub(crate) fn prepare(
        pool: &MemoryLedger,
        shape: MaterializationPayloadShape,
    ) -> Result<Self, ColdMaterializationSlotError> {
        pool.initialize_shared_native(Initializer(shape))
            .map(Self)
            .map_err(ColdMaterializationSlotError)
    }

    /// A foreign pool leaves the unissued slot intact. Successful checkout can
    /// happen only once and creates neither another node nor request authority.
    pub(crate) fn take(
        &mut self,
        pool: &MemoryLedger,
    ) -> Result<PreparedWeightMaterialization, WorkingMemoryError> {
        self.0.validate_pool(pool)?;
        self.0
            .output()
            .0
            .borrow_mut()
            .take()
            .ok_or(WorkingMemoryError::IdentityMismatch)
    }
}

#[cfg(test)]
mod tests;
