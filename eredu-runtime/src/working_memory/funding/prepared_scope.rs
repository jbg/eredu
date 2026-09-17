//! A native funding scope whose funding interface has never been exposed.
use super::*;
use std::mem::size_of;

/// Same original counted scope, without adoption, capture, or certification
/// access. Dropping unused custody closes only this never-exposed scope. It
/// does not establish native completion or clear another owner's quarantine.
#[derive(Debug)]
#[must_use]
pub struct PreparedWorkingMemoryFundingScope {
    scope: Option<WorkingMemoryFundingScope>,
}

impl WorkingMemoryFundingRun {
    /// Counts the same native scope as `scope`, while withholding its funding
    /// interface until consumption. An unused owner can cancel its own count.
    pub fn prepare_scope(&self) -> Result<PreparedWorkingMemoryFundingScope, WorkingMemoryError> {
        Ok(PreparedWorkingMemoryFundingScope {
            scope: Some(self.scope_with_purpose(ScopePurpose::Native)?),
        })
    }
}

impl PreparedWorkingMemoryFundingScope {
    /// Moves the exact original scope into its active interface. This performs
    /// no allocation, accounting, validation, callback, or completion action.
    /// Native composition places this before its first worker operation.
    pub fn activate(mut self) -> WorkingMemoryFundingScope {
        self.scope
            .take()
            .expect("unconsumed prepared funding scope")
    }

    pub(in crate::working_memory) fn native(&self) -> &WorkingMemoryFundingScope {
        self.scope
            .as_ref()
            .expect("unconsumed prepared funding scope")
    }

    /// Concrete factory/extraction/cancellation control representations only.
    /// The retained wrapper is measured by its enclosing owner. This includes
    /// the existing canonical poison-prefix node for this one prepared scope;
    /// it creates no allowance or general native-scope population bound.
    pub fn control_bytes() -> Option<usize> {
        [
            size_of::<Self>(), // factory return and consuming activation argument
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<WorkingMemoryFundingScope>(), // original factory/activation move
            size_of::<Option<WorkingMemoryFundingScope>>(), // extraction
            size_of::<std::sync::LockResult<std::sync::MutexGuard<'_, Usage>>>(),
            size_of::<std::sync::MutexGuard<'_, Usage>>(),
            size_of::<QuarantinedStoragePins>(), // existing poison Drop allocation
            size_of::<Box<QuarantinedStoragePins>>(), // node construction/link move
            size_of::<Option<Box<QuarantinedStoragePins>>>(),
            size_of::<Option<RegisteredStoragePin>>(), // borrowed pin taken before Usage
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}

impl Drop for PreparedWorkingMemoryFundingScope {
    fn drop(&mut self) {
        let Some(scope) = self.scope.as_mut() else {
            return;
        };
        // No funding interface has escaped this owner. Under a healthy loan,
        // close only its own count. Existing quarantine and stamped exclusions
        // remain in place and settle continues to retain their envelope.
        // On poison, leave active=true. The failed loan is gone before field
        // Drop reaches the ordinary pin-preserving quarantine path.
        if let Ok(mut usage) = scope.pool.0.usage.lock() {
            let state = usage
                .funding
                .get_mut(&scope.id)
                .expect("live prepared scope");
            state.close_scope(scope.purpose);
            scope.active = false;
            settle(&mut usage, scope.id);
            drop(usage);
            accounts::drain(&scope.pool);
        }
        // All borrowed provider pins and pool owners are fields of scope and
        // drop after the Usage loan. No synthetic Status or certify call.
    }
}
