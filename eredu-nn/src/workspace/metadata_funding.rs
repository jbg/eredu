//! Workspace compatibility surface over the shared neutral metadata account.
pub use eredu_core::{
    HostMetadataAccount as WorkspaceMetadataAccount,
    HostMetadataFundingError as WorkspaceMetadataFundingError,
};
use std::{fmt, ops::Deref};

/// The same core account with the workspace's counted vector/text/error helpers.
/// This transparent handle adds no allocation, reservation or retirement owner.
/// Conversions preserve the actual cumulative account and all its existing aliases.
#[repr(transparent)]
#[derive(Clone)]
pub struct WorkspaceMetadataFunding(eredu_core::HostMetadataFunding);
impl WorkspaceMetadataFunding {
    /// Exact same core account Box, shared shell and constructor transports.
    pub fn constructor_bytes<A: WorkspaceMetadataAccount>() -> Option<usize> {
        eredu_core::HostMetadataFunding::constructor_bytes::<A>()
    }
    /// Pays and constructs the sole shared core account before either allocation.
    pub fn new<A: WorkspaceMetadataAccount>(account: A) -> Result<Self, WorkspaceMetadataFundingError> {
        eredu_core::HostMetadataFunding::new(account).map(Self)
    }
}
impl Deref for WorkspaceMetadataFunding {
    type Target = eredu_core::HostMetadataFunding;
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl From<eredu_core::HostMetadataFunding> for WorkspaceMetadataFunding {
    fn from(funding: eredu_core::HostMetadataFunding) -> Self { Self(funding) }
}
impl From<WorkspaceMetadataFunding> for eredu_core::HostMetadataFunding {
    fn from(funding: WorkspaceMetadataFunding) -> Self { funding.0 }
}
impl fmt::Debug for WorkspaceMetadataFunding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("WorkspaceMetadataFunding").finish_non_exhaustive()
    }
}
pub(super) const fn reservation_control_bytes() -> usize {
    eredu_core::HostMetadataFunding::reservation_control_bytes()
}

#[cfg(test)]
mod tests;
