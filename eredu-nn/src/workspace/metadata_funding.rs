//! Canonical neutral metadata account reexported for workspace consumers.
pub use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};

pub(super) const fn reservation_control_bytes() -> usize {
    HostMetadataFunding::reservation_control_bytes()
}

#[cfg(test)]
mod tests;
