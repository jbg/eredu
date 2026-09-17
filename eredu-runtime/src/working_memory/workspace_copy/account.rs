//! Constructor controls for the existing independently admitted copy account.
use super::*;
use eredu_core::HostPreparationAuthority;
use std::mem::size_of;

/// Finite account-constructor contribution to an enclosing host preparation.
/// This prices no numerical copy, table payload, source pin or native work.
/// Querying or consuming it never raises a copy budget or creates a scope.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceCopyAccountLayout {
    bytes: usize,
}
impl WorkspaceCopyAccountLayout {
    /// One native copy scope, without sampler or decoder host scopes.
    pub fn workspace() -> Result<Self, WorkingMemoryError> {
        Self::new(false, 0, false)
    }
    /// Same native scope plus the fixed pair retaining an actual original B source.
    pub fn workspace_with_prepared_source() -> Result<Self, WorkingMemoryError> {
        let mut layout = Self::workspace()?;
        layout.bytes = layout
            .bytes
            .checked_add(RegisteredStoragePin::pair_control_bytes(true)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(layout)
    }
    /// One sampler host scope and the separate native copy scope.
    pub fn sampling() -> Result<Self, WorkingMemoryError> {
        Self::new(true, 0, false)
    }
    /// One sampler scope, one actual decoder table scope and native scope.
    pub fn decoder_table() -> Result<Self, WorkingMemoryError> {
        Self::new(true, 1, false)
    }
    /// One sampler scope, every actual outer/child table scope, and native scope.
    /// Group Vec payloads/constructors are covered by the group table producer.
    pub fn decoder_group(tables: usize) -> Result<Self, WorkingMemoryError> {
        if tables == 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Self::new(true, tables, true)
    }
    fn new(sampler: bool, tables: usize, group: bool) -> Result<Self, WorkingMemoryError> {
        let bytes = [
            super::super::funding::copy_account_control_bytes(sampler, tables, group)?,
            usize::try_from(super::super::qualified_storage::shared_bytes::<()>()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            usize::try_from(super::super::qualified_storage::shared_bytes::<CopyAccount>()?)
                .map_err(|_| WorkingMemoryError::Overflow)?,
            size_of::<CopyAccount>(), // actual shared value construction
            size_of::<Option<CopyAccount>>(), // final Arc::into_inner transport
            size_of::<std::sync::Arc<CopyAccount>>(),
            size_of::<Option<std::sync::Arc<CopyAccount>>>(),
            size_of::<WorkspaceCopyRetention>(),
            size_of::<Self>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<InferenceExecutionIdentity>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Option<HostPreparationAuthority>>(),
            size_of::<WorkspaceCopyCustody>(),
            size_of::<AdmittedWorkspaceCopy>(),
            size_of::<(WorkspaceCopyCustody, WorkingMemoryFundingScope)>(),
            size_of::<Result<AdmittedWorkspaceCopy, WorkspaceCopyAdmissionError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { bytes })
    }
    /// Known H contribution only. The enclosing source-bound host plan admits
    /// this amount before any account or identity constructor runs.
    pub fn requested_bytes(self) -> usize {
        self.bytes
    }

    pub(in crate::working_memory) fn execution(
        self,
        preparation: &HostPreparationAuthority,
    ) -> InferenceExecutionIdentity {
        // Construction runs only from the actual prepared source carrier. The
        // independent B account still validates sources and exact copy limits.
        InferenceExecutionIdentity(std::sync::Arc::new(()), Some(preparation.clone()))
    }
}
