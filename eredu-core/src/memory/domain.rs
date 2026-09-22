//! Physical placement and checked, topology-qualified memory requirements.
//!
//! These descriptors confer no allocation, source, or execution authority.
//! Backends declare physical sharing; allocation identities establish sharing
//! of storage. A location mapping never makes two allocations into one.

use std::sync::atomic::{AtomicU64, Ordering};

mod limits;
mod placement;
mod requirements;
pub use limits::{MemoryHeadroomDeclarations, MemoryLimit, MemoryLimitDeclarations, MemoryLimits};
pub use placement::{MemoryPlacement, MemoryPlacementKind};
pub use requirements::{
    DomainMemoryCharge, DomainMemoryRequirements, DomainOverheadEstimate, PlacementAllowance,
};

/// A physical domain within one immutable, process-local topology.
///
/// Even identically described independent topologies have different identities.
/// Identifiers cannot be deserialized into admission authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryDomainId {
    topology: u64,
    slot: usize,
}

/// Backend-qualified device location; the ordinal alone is not an identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryDeviceId {
    /// Stable backend namespace, such as `mlx`.
    pub backend: &'static str,
    /// Device ordinal within that backend's registered process-local devices.
    pub ordinal: u32,
}

/// A location resolved to physical memory using backend facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryLocation {
    /// Process host memory. NUMA-specific budgets are outside this contract.
    Host,
    /// One registered accelerator.
    Device(MemoryDeviceId),
}

/// Backend declaration of one domain and its physically sharing locations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDomainDescription {
    /// Stable identifier used by configuration and diagnostics, e.g. `host`.
    pub name: String,
    /// Nonempty locations established to share this domain's physical memory.
    pub locations: Vec<MemoryLocation>,
}

/// Immutable physical domains and location mapping for one ledger.
///
/// Construction validates declarations but cannot authenticate hardware facts.
/// A backend must obtain those facts from its selected allocation mechanisms.
/// Host accessibility, feature selection, and matching ordinals do not prove
/// physical sharing. Moving the topology preserves identity; constructing an
/// identical topology creates a distinct identity.
#[derive(Debug)]
pub struct MemoryTopology {
    identity: u64,
    domains: Vec<MemoryDomainDescription>,
    host: usize,
}

impl MemoryTopology {
    /// Validates a complete process-local declaration, including exactly one
    /// host location. No native resources are created or queried here.
    pub fn new(domains: Vec<MemoryDomainDescription>) -> Result<Self, MemoryDomainError> {
        if domains.is_empty() {
            return Err(MemoryDomainError::EmptyTopology);
        }
        let mut host = None;
        for (slot, domain) in domains.iter().enumerate() {
            if domain.name.is_empty()
                || !domain
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            {
                return Err(MemoryDomainError::InvalidDomainName { declaration: slot });
            }
            if domains[..slot]
                .iter()
                .any(|prior| prior.name == domain.name)
            {
                return Err(MemoryDomainError::DuplicateDomainName { declaration: slot });
            }
            if domain.locations.is_empty() {
                return Err(MemoryDomainError::EmptyLocations { declaration: slot });
            }
            for (index, location) in domain.locations.iter().enumerate() {
                if let MemoryLocation::Device(device) = location {
                    if device.backend.trim().is_empty() {
                        return Err(MemoryDomainError::InvalidDevice);
                    }
                }
                if domain.locations[..index].contains(location)
                    || domains[..slot]
                        .iter()
                        .any(|prior| prior.locations.contains(location))
                {
                    return Err(MemoryDomainError::DuplicateLocation {
                        location: *location,
                    });
                }
                if *location == MemoryLocation::Host {
                    host = Some(slot);
                }
            }
        }
        let host = host.ok_or(MemoryDomainError::MissingHost)?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let identity = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| MemoryDomainError::Overflow)?;
        Ok(Self {
            identity,
            domains,
            host,
        })
    }

    /// Domains in their fixed dense slot order.
    pub fn domains(
        &self,
    ) -> impl ExactSizeIterator<Item = (MemoryDomainId, &MemoryDomainDescription)> {
        self.domains
            .iter()
            .enumerate()
            .map(|(slot, description)| (self.id(slot), description))
    }

    /// Number of physical domains, independent of the number of locations.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    /// A valid topology always contains its host domain.
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }

    /// Domain for process host allocations, including ledger metadata.
    pub fn host_domain(&self) -> MemoryDomainId {
        self.id(self.host)
    }

    /// Resolves a backend-established location; absent devices are errors.
    pub fn domain_for(
        &self,
        location: MemoryLocation,
    ) -> Result<MemoryDomainId, MemoryDomainError> {
        self.domains
            .iter()
            .position(|domain| domain.locations.contains(&location))
            .map(|slot| self.id(slot))
            .ok_or(MemoryDomainError::UnknownLocation { location })
    }

    /// Resolves a physical identifier for CLI or application configuration.
    pub fn domain_named(&self, name: &str) -> Option<MemoryDomainId> {
        self.domains
            .iter()
            .position(|domain| domain.name == name)
            .map(|slot| self.id(slot))
    }

    /// Validates identity before exposing a dense slot to a ledger.
    pub fn slot(&self, domain: MemoryDomainId) -> Result<usize, MemoryDomainError> {
        if domain.topology != self.identity {
            return Err(MemoryDomainError::ForeignTopology);
        }
        if domain.slot >= self.domains.len() {
            return Err(MemoryDomainError::UnknownDomain { domain });
        }
        Ok(domain.slot)
    }

    /// Description belonging to this exact topology.
    pub fn description(
        &self,
        domain: MemoryDomainId,
    ) -> Result<&MemoryDomainDescription, MemoryDomainError> {
        Ok(&self.domains[self.slot(domain)?])
    }

    /// Controlled backing capacities owned by the topology. The containing
    /// `MemoryTopology` representation is priced by its owner separately.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        let mut bytes = vector_bytes::<MemoryDomainDescription>(self.domains.capacity())?;
        for domain in &self.domains {
            bytes = bytes
                .checked_add(vector_bytes::<u8>(domain.name.capacity())?)
                .and_then(|bytes| {
                    bytes.checked_add(
                        vector_bytes::<MemoryLocation>(domain.locations.capacity()).ok()?,
                    )
                })
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }

    fn id(&self, slot: usize) -> MemoryDomainId {
        MemoryDomainId {
            topology: self.identity,
            slot,
        }
    }
}

fn vector_bytes<T>(capacity: usize) -> Result<u64, MemoryDomainError> {
    capacity
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(MemoryDomainError::Overflow)
}

/// Fixed, allocation-free topology, declaration, or capacity failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MemoryDomainError {
    /// At least the host domain is required.
    #[error("memory topology has no domains")]
    EmptyTopology,
    /// Domain names use only ASCII letters, digits, underscores and hyphens.
    #[error("invalid physical memory domain name at declaration {declaration}")]
    InvalidDomainName {
        /// Declaration index.
        declaration: usize,
    },
    /// Different domains cannot have the same configuration identifier.
    #[error("duplicate physical memory domain name at declaration {declaration}")]
    DuplicateDomainName {
        /// Declaration index.
        declaration: usize,
    },
    /// Every domain must have at least one physical location.
    #[error("physical memory domain {declaration} has no locations")]
    EmptyLocations {
        /// Declaration index.
        declaration: usize,
    },
    /// Device namespace was empty.
    #[error("memory device has an empty backend namespace")]
    InvalidDevice,
    /// Host and device locations each have exactly one physical domain.
    #[error("memory location {location:?} was declared more than once")]
    DuplicateLocation {
        /// Repeated location.
        location: MemoryLocation,
    },
    /// Host bookkeeping cannot be placed without the host domain.
    #[error("memory topology has no host location")]
    MissingHost,
    /// A location is absent from the immutable topology.
    #[error("memory location {location:?} is not registered")]
    UnknownLocation {
        /// Missing location.
        location: MemoryLocation,
    },
    /// Descriptors from another topology cannot be combined.
    #[error("memory descriptor belongs to another topology")]
    ForeignTopology,
    /// The domain is absent from this topology.
    #[error("unknown physical memory domain {domain:?}")]
    UnknownDomain {
        /// Invalid domain.
        domain: MemoryDomainId,
    },
    /// More than one limit was supplied for a domain, even if values agree.
    #[error("duplicate memory limit for physical domain {domain:?}")]
    DuplicateLimit {
        /// Repeated domain.
        domain: MemoryDomainId,
    },
    /// Configuration names must resolve to a physical domain.
    #[error("unknown physical memory domain at limit declaration {declaration}")]
    UnknownDomainName {
        /// Declaration index.
        declaration: usize,
    },
    /// A candidate set must be finite, nonempty, and have a stated basis.
    #[error("memory placement requires candidates and a nonempty estimation basis")]
    InvalidPlacement,
    /// Finite estimates must identify their source and derivation.
    #[error("finite memory estimate requires a source and estimation basis")]
    InvalidEstimateDescription,
    /// A descriptive component cannot be removed unless its complete provenance
    /// and per-domain categories are present in the composed requirements.
    #[error("memory requirements do not contain the complete described component")]
    MissingRequirementComponent,
    /// A registered domain was not in this allocation's candidate set.
    #[error("physical memory domain {domain:?} is outside the admitted placement")]
    UnaccountedAccess {
        /// Domain the operation would access.
        domain: MemoryDomainId,
    },
    /// Capacity failed after checked computation of the complete live charge.
    #[error("memory domain {domain:?}: {existing_bytes} existing + {requested_bytes} requested bytes exceeds finite limit {limit_bytes}")]
    BudgetExceeded {
        /// Failing physical domain.
        domain: MemoryDomainId,
        /// Finite effective limit; `u64::MAX` remains finite.
        limit_bytes: u64,
        /// Charge already live in the domain.
        existing_bytes: u64,
        /// Proposed incremental charge.
        requested_bytes: u64,
    },
    /// Every amount and identifier calculation remains checked under unlimited limits.
    #[error("physical memory accounting arithmetic overflow")]
    Overflow,
}

#[cfg(test)]
mod tests;
