use super::*;

/// Physical placement descriptor, validated against an immutable topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlacement {
    pub(super) kind: MemoryPlacementKind,
}

/// Exact physical placement or conservative candidate-domain allowances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPlacementKind {
    /// Allocation charged once in its known physical domain.
    Fixed(MemoryDomainId),
    /// Full backing capacity is allowed for in each distinct candidate domain.
    /// These allowances are not measured residency and never follow telemetry.
    Possible {
        /// Nonempty distinct candidate domains in topology slot order.
        domains: Vec<MemoryDomainId>,
        /// Source-labelled basis supplied by the allocation mechanism.
        basis: String,
    },
}

impl MemoryPlacement {
    /// Known physical domain, independent of the stream accessing an alias.
    pub fn fixed(
        topology: &MemoryTopology,
        domain: MemoryDomainId,
    ) -> Result<Self, MemoryDomainError> {
        topology.slot(domain)?;
        Ok(Self {
            kind: MemoryPlacementKind::Fixed(domain),
        })
    }

    /// Conservative full-capacity allowances in all supplied candidate domains.
    /// Duplicate locations/domains collapse, but independent allocations do not.
    /// Even a single candidate remains a labelled allowance rather than telemetry.
    pub fn possible(
        topology: &MemoryTopology,
        mut domains: Vec<MemoryDomainId>,
        basis: String,
    ) -> Result<Self, MemoryDomainError> {
        if domains.is_empty() || basis.trim().is_empty() {
            return Err(MemoryDomainError::InvalidPlacement);
        }
        for &domain in &domains {
            topology.slot(domain)?;
        }
        domains.sort_unstable();
        domains.dedup();
        Ok(Self {
            kind: MemoryPlacementKind::Possible { domains, basis },
        })
    }

    /// Resolves a finite backend-supplied candidate location set before summing.
    pub fn possible_locations(
        topology: &MemoryTopology,
        locations: impl IntoIterator<Item = MemoryLocation>,
        basis: String,
    ) -> Result<Self, MemoryDomainError> {
        let domains = locations
            .into_iter()
            .map(|location| topology.domain_for(location))
            .collect::<Result<Vec<_>, _>>()?;
        Self::possible(topology, domains, basis)
    }

    /// Borrowed placement; mutation cannot invalidate the candidate set.
    pub fn kind(&self) -> &MemoryPlacementKind {
        &self.kind
    }

    /// Physical domains charged by this allocation.
    pub fn domains(&self) -> &[MemoryDomainId] {
        match &self.kind {
            MemoryPlacementKind::Fixed(domain) => std::slice::from_ref(domain),
            MemoryPlacementKind::Possible { domains, .. } => domains,
        }
    }

    /// Reject access to a domain absent from the admitted candidate set before
    /// launching native work. This check grants no submission authority itself.
    pub fn validate_access(
        &self,
        topology: &MemoryTopology,
        location: MemoryLocation,
    ) -> Result<(), MemoryDomainError> {
        self.validate(topology)?;
        let domain = topology.domain_for(location)?;
        if !self.domains().contains(&domain) {
            return Err(MemoryDomainError::UnaccountedAccess { domain });
        }
        Ok(())
    }

    /// Checks placement identity without consulting current residency telemetry.
    pub fn validate(&self, topology: &MemoryTopology) -> Result<(), MemoryDomainError> {
        for &domain in self.domains() {
            topology.slot(domain)?;
        }
        Ok(())
    }

    /// Controlled descriptor backing, separately charged to host metadata.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        match &self.kind {
            MemoryPlacementKind::Fixed(_) => Ok(0),
            MemoryPlacementKind::Possible { domains, basis } => {
                vector_bytes::<MemoryDomainId>(domains.capacity())?
                    .checked_add(vector_bytes::<u8>(basis.capacity())?)
                    .ok_or(MemoryDomainError::Overflow)
            }
        }
    }

    /// Backing requested by a deep clone, excluding spare source capacity.
    pub fn clone_backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        match &self.kind {
            MemoryPlacementKind::Fixed(_) => Ok(0),
            MemoryPlacementKind::Possible { domains, basis } => {
                vector_bytes::<MemoryDomainId>(domains.len())?
                    .checked_add(vector_bytes::<u8>(basis.len())?)
                    .ok_or(MemoryDomainError::Overflow)
            }
        }
    }
}
