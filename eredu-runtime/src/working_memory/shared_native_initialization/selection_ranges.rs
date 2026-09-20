//! Admission for the checkpoint-owned selection range destination.
use super::*;
use eredu_checkpoint::store::{SelectionReadDestinationPlan, SelectionReadRanges};

impl SharedNativeInitializer for SelectionReadDestinationPlan<'_, '_> {
    type Output = SelectionReadRanges;
    type Error = std::collections::TryReserveError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.required_bytes().ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        _custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        // All allocation and fill work completes synchronously. The existing
        // initialization owner retains the result's account; no alias escapes.
        self.build()
    }
}

#[cfg(test)]
mod tests;
