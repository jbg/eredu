//! Exact independently addressable native storage facilities.

/// Generic independently addressable storage facilities implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AddressableStorageCapabilities {
    bulk_access: bool,
    incremental_access: bool,
    lease_completion: bool,
    maximum_compact_bytes: u64,
    tiers: AddressableStorageTiers,
}

/// Generic storage tiers usable by an independently addressable bank.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AddressableStorageTiers {
    device: bool,
    host: bool,
    disk: bool,
}

impl AddressableStorageTiers {
    /// Creates one exact tier capability set.
    pub const fn new(device: bool, host: bool, disk: bool) -> Self {
        Self { device, host, disk }
    }

    /// Returns whether executable device storage is available.
    pub const fn device(self) -> bool {
        self.device
    }

    /// Returns whether host staging storage is available.
    pub const fn host(self) -> bool {
        self.host
    }

    /// Returns whether lazy checkpoint-backed storage is available.
    pub const fn disk(self) -> bool {
        self.disk
    }
}

impl AddressableStorageCapabilities {
    /// Creates an exact addressable-storage capability report.
    pub const fn new(
        bulk_access: bool,
        incremental_access: bool,
        lease_completion: bool,
        maximum_compact_bytes: u64,
    ) -> Self {
        Self {
            bulk_access,
            incremental_access,
            lease_completion,
            maximum_compact_bytes,
            tiers: AddressableStorageTiers::new(true, true, true),
        }
    }

    /// Replaces the exact supported storage-tier set.
    pub const fn with_tiers(mut self, tiers: AddressableStorageTiers) -> Self {
        self.tiers = tiers;
        self
    }

    /// Returns whether bounded multi-row access is implemented.
    pub const fn bulk_access(self) -> bool {
        self.bulk_access
    }

    /// Returns whether latency-sensitive incremental access is implemented.
    pub const fn incremental_access(self) -> bool {
        self.incremental_access
    }

    /// Returns whether acquisitions remain leased through native completion.
    pub const fn lease_completion(self) -> bool {
        self.lease_completion
    }

    /// Returns the largest supported per-acquisition compact bank.
    pub const fn maximum_compact_bytes(self) -> u64 {
        self.maximum_compact_bytes
    }

    /// Returns the exact independently addressable storage tiers.
    pub const fn tiers(self) -> AddressableStorageTiers {
        self.tiers
    }
}
