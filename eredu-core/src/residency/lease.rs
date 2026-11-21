//! Move-only authority for one resident copy.

use std::sync::Weak;

use super::{MemoryTier, OffloadUnitId};

/// Backend-native host/device storage exposed through a neutral residency lease.
pub trait ResidencyLeaseStorage {
    /// Backend-native executable value.
    type DeviceValue;
    /// Backend-native host-resident value.
    type HostValue;
    /// Concrete lookup failure.
    type Error;
    /// Allocation-free or cold-path iterator over stable binding names.
    type BindingNames<'a>: Iterator<Item = &'a str>
    where
        Self: 'a;

    /// Looks up one executable binding.
    fn device_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::DeviceValue, Self::Error>;

    /// Looks up one host-resident binding.
    fn host_value<'a>(
        &'a self,
        id: &OffloadUnitId,
        name: &str,
    ) -> Result<&'a Self::HostValue, Self::Error>;

    /// Returns binding names in stable order.
    fn binding_names(&self) -> Self::BindingNames<'_>;
}

/// Concrete manager hook used when a residency lease releases its exact pin.
pub trait ResidencyLeaseOwner: Send + Sync {
    /// Releases one tier pin. Drop paths must tolerate an already-destroyed manager.
    fn release_residency_pin(&self, id: &OffloadUnitId, tier: MemoryTier);
}

/// Move-only publication authority retained alongside native transfer resources.
pub trait ResidencyTransferTicket {
    /// Failure publishing or retiring the selected copy.
    type Error;

    /// Records the exact terminal transfer disposition. Call only once native
    /// work has completed or reached safe failure or teardown. Polling errors
    /// alone do not authorize settlement.
    fn settle(&mut self, succeeded: bool) -> Result<(), Self::Error>;
}

/// Lease retaining one native resident unit and an opaque residency pin.
pub struct ResidencyLease<S>
where
    S: ResidencyLeaseStorage,
{
    id: OffloadUnitId,
    tier: MemoryTier,
    storage: S,
    owner: Weak<dyn ResidencyLeaseOwner>,
}

impl<S> ResidencyLease<S>
where
    S: ResidencyLeaseStorage,
{
    /// Creates a lease after the neutral controller has pinned the resident copy.
    pub fn new(
        id: OffloadUnitId,
        tier: MemoryTier,
        storage: S,
        owner: Weak<dyn ResidencyLeaseOwner>,
    ) -> Self {
        Self {
            id,
            tier,
            storage,
            owner,
        }
    }

    /// Returns the acquired unit identifier.
    pub fn id(&self) -> &OffloadUnitId {
        &self.id
    }

    /// Returns the protected resident tier.
    pub const fn tier(&self) -> MemoryTier {
        self.tier
    }

    /// Looks up one backend-native executable binding without cloning it.
    pub fn device_value(&self, name: &str) -> Result<&S::DeviceValue, S::Error> {
        self.storage.device_value(&self.id, name)
    }

    /// Looks up one backend-native host binding without cloning it.
    pub fn host_value(&self, name: &str) -> Result<&S::HostValue, S::Error> {
        self.storage.host_value(&self.id, name)
    }

    /// Returns binding names in stable order.
    pub fn binding_names(&self) -> S::BindingNames<'_> {
        self.storage.binding_names()
    }
}

impl<S> Drop for ResidencyLease<S>
where
    S: ResidencyLeaseStorage,
{
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner.release_residency_pin(&self.id, self.tier);
        }
    }
}
