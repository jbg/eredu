//! Borrowed JSON producer policy, retaining the first actual host refusal.
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use serde_json::allocation::{Allocation, AllocationError};
use std::{cell::Cell, mem::size_of};
pub(super) struct JsonAllocation<'a> {
    funding: &'a HostMetadataFunding,
    failure: Cell<Option<HostMetadataFundingError>>,
}
impl<'a> JsonAllocation<'a> {
    pub(super) fn new(funding: &'a HostMetadataFunding) -> Result<Self, HostMetadataFundingError> {
        funding.reserve_metadata(
            size_of::<Self>()
                + size_of::<Result<Self, HostMetadataFundingError>>()
                + HostMetadataFunding::reservation_control_bytes(),
        )?;
        Ok(Self {
            funding,
            failure: Cell::new(None),
        })
    }
    pub(super) fn failure(&self) -> Option<HostMetadataFundingError> {
        self.failure.get()
    }
}
impl Allocation for JsonAllocation<'_> {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        if self.failure.get().is_some() {
            return Err(AllocationError::Refused);
        }
        self.funding.reserve_metadata(bytes).map_err(|error| {
            self.failure.set(Some(error));
            AllocationError::Refused
        })
    }
}
