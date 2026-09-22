use super::*;

/// Explicit finite or unlimited capacity; there is no numeric infinity sentinel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLimit {
    /// Inclusive maximum charge in bytes.
    Finite(u64),
    /// Skip capacity comparison, while preserving checked arithmetic and evidence.
    #[default]
    Unlimited,
}

/// Named physical-domain limits supplied before a backend topology is resolved.
///
/// Empty declarations mean `Unlimited` in every domain. Resolution authenticates
/// every name and rejects duplicate declarations; names never identify storage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryLimitDeclarations(Option<std::sync::Arc<[(String, MemoryLimit)]>>);

/// Additional capacity allowances attributed to named physical domains.
/// Headroom remains separate from accounted allocations and finite estimates;
/// it does not make unknown overhead finite.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryHeadroomDeclarations(Option<std::sync::Arc<[(String, u64)]>>);

impl MemoryHeadroomDeclarations {
    /// Creates headroom declarations for resolution before admission.
    pub fn new(declarations: impl IntoIterator<Item = (String, u64)>) -> Self {
        let declarations: Vec<_> = declarations.into_iter().collect();
        Self((!declarations.is_empty()).then(|| declarations.into()))
    }

    /// No additional headroom is requested.
    pub const fn none() -> Self {
        Self(None)
    }

    /// Named increments in application order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, u64)> + '_ {
        self.0
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|(name, bytes)| (name.as_str(), *bytes))
    }

    /// Checked requirements preserving each increment's physical domain.
    /// Repeated names add their allowances with checked arithmetic.
    pub fn resolve(
        &self,
        topology: &MemoryTopology,
    ) -> Result<DomainMemoryRequirements, MemoryDomainError> {
        let mut requirements = DomainMemoryRequirements::zero(topology);
        for (declaration, (name, bytes)) in self.iter().enumerate() {
            let domain = topology
                .domain_named(name)
                .ok_or(MemoryDomainError::UnknownDomainName { declaration })?;
            requirements.add_headroom(domain, bytes)?;
        }
        Ok(requirements)
    }

    /// Controlled configuration backing, excluding the owner's inline value.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        let Some(declarations) = &self.0 else {
            return Ok(0);
        };
        let mut bytes = vector_bytes::<(String, u64)>(declarations.len())?
            .checked_add(vector_bytes::<usize>(2)?)
            .ok_or(MemoryDomainError::Overflow)?;
        for (name, _) in declarations.iter() {
            bytes = bytes
                .checked_add(vector_bytes::<u8>(name.capacity())?)
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }
}

impl MemoryLimitDeclarations {
    /// Creates an unresolved configuration. Resolve it before admission.
    pub fn new(declarations: impl IntoIterator<Item = (String, MemoryLimit)>) -> Self {
        let declarations: Vec<_> = declarations.into_iter().collect();
        Self((!declarations.is_empty()).then(|| declarations.into()))
    }

    /// The default policy imposes no additional per-domain ceiling.
    pub const fn unlimited() -> Self {
        Self(None)
    }

    /// Named declarations in application order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, MemoryLimit)> + '_ {
        self.0
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|(name, limit)| (name.as_str(), *limit))
    }

    /// Complete authenticated limits for the backend's immutable topology.
    pub fn resolve(&self, topology: &MemoryTopology) -> Result<MemoryLimits, MemoryDomainError> {
        MemoryLimits::resolve_named(topology, self.iter())
    }

    /// Controlled configuration backing, excluding the owner's inline value.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        let Some(declarations) = &self.0 else {
            return Ok(0);
        };
        let mut bytes = vector_bytes::<(String, MemoryLimit)>(declarations.len())?
            .checked_add(vector_bytes::<usize>(2)?)
            .ok_or(MemoryDomainError::Overflow)?;
        for (name, _) in declarations.iter() {
            bytes = bytes
                .checked_add(vector_bytes::<u8>(name.capacity())?)
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }
}

macro_rules! declarations_serialization {
    ($name:ident, $value:ty) => {
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serde::Serialize::serialize(self.0.as_deref().unwrap_or_default(), serializer)
            }
        }
        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let values =
                    <Vec<(String, $value)> as serde::Deserialize>::deserialize(deserializer)?;
                Ok(Self::new(values))
            }
        }
        impl $name {
            /// Whether these values retain the same immutable allocation owner.
            pub fn same_owner(&self, other: &Self) -> bool {
                match (&self.0, &other.0) {
                    (None, None) => true,
                    (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
                    _ => false,
                }
            }
        }
    };
}
declarations_serialization!(MemoryLimitDeclarations, MemoryLimit);
declarations_serialization!(MemoryHeadroomDeclarations, u64);

impl MemoryLimit {
    /// Intersection of configured and live account constraints.
    pub const fn minimum(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unlimited, limit) | (limit, Self::Unlimited) => limit,
            (Self::Finite(a), Self::Finite(b)) => Self::Finite(if a < b { a } else { b }),
        }
    }

    /// Whether adopting `next` relaxes this constraint. Raising a live account's
    /// limit requires separate move-only succession authority from its owner.
    pub const fn is_raised_by(self, next: Self) -> bool {
        match (self, next) {
            (Self::Finite(a), Self::Finite(b)) => b > a,
            (Self::Finite(_), Self::Unlimited) => true,
            (Self::Unlimited, _) => false,
        }
    }

    /// Checks arithmetic before capacity, including under `Unlimited`.
    pub fn check(
        self,
        domain: MemoryDomainId,
        existing_bytes: u64,
        requested_bytes: u64,
    ) -> Result<u64, MemoryDomainError> {
        let total = existing_bytes
            .checked_add(requested_bytes)
            .ok_or(MemoryDomainError::Overflow)?;
        if let Self::Finite(limit_bytes) = self {
            if total > limit_bytes {
                return Err(MemoryDomainError::BudgetExceeded {
                    domain,
                    limit_bytes,
                    existing_bytes,
                    requested_bytes,
                });
            }
        }
        Ok(total)
    }
}

/// Complete, immutable limit vector for an exact topology.
///
/// Request limits constrain the total live charge, including existing storage
/// and other work. Unspecified domains resolve to `Unlimited`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLimits {
    topology: u64,
    limits: Vec<MemoryLimit>,
}

impl MemoryLimits {
    /// Reuses an already funded dense vector to resolve named declarations.
    /// Invalid names and duplicate declarations leave the vector unchanged.
    pub fn resolve_named_in_place(
        &mut self,
        topology: &MemoryTopology,
        declarations: &MemoryLimitDeclarations,
        ceiling: Option<&Self>,
    ) -> Result<(), MemoryDomainError> {
        self.validate(topology)?;
        if let Some(ceiling) = ceiling {
            ceiling.validate(topology)?;
        }
        for (index, (name, _)) in declarations.iter().enumerate() {
            let domain = topology
                .domain_named(name)
                .ok_or(MemoryDomainError::UnknownDomainName { declaration: index })?;
            if declarations
                .iter()
                .take(index)
                .any(|(prior, _)| prior == name)
            {
                return Err(MemoryDomainError::DuplicateLimit { domain });
            }
        }
        for (slot, (domain, description)) in topology.domains().enumerate() {
            let declared = declarations
                .iter()
                .find(|(name, _)| *name == description.name)
                .map_or(MemoryLimit::Unlimited, |(_, limit)| limit);
            self.limits[slot] = declared.minimum(ceiling.map_or(MemoryLimit::Unlimited, |value| {
                value.get(domain).expect("validated topology")
            }));
        }
        Ok(())
    }

    /// Resolves declarations, rejecting duplicates and foreign domain identities.
    pub fn resolve(
        topology: &MemoryTopology,
        declarations: impl IntoIterator<Item = (MemoryDomainId, MemoryLimit)>,
    ) -> Result<Self, MemoryDomainError> {
        let mut limits = vec![None; topology.len()];
        for (domain, limit) in declarations {
            let slot = topology.slot(domain)?;
            if limits[slot].replace(limit).is_some() {
                return Err(MemoryDomainError::DuplicateLimit { domain });
            }
        }
        Ok(Self {
            topology: topology.identity,
            limits: limits
                .into_iter()
                .map(|limit| limit.unwrap_or_default())
                .collect(),
        })
    }

    /// Resolves CLI/application identifiers to physical domains before admission.
    pub fn resolve_named<'a>(
        topology: &MemoryTopology,
        declarations: impl IntoIterator<Item = (&'a str, MemoryLimit)>,
    ) -> Result<Self, MemoryDomainError> {
        let mut limits = vec![None; topology.len()];
        for (declaration, (name, limit)) in declarations.into_iter().enumerate() {
            let domain = topology
                .domain_named(name)
                .ok_or(MemoryDomainError::UnknownDomainName { declaration })?;
            if limits[domain.slot].replace(limit).is_some() {
                return Err(MemoryDomainError::DuplicateLimit { domain });
            }
        }
        Ok(Self {
            topology: topology.identity,
            limits: limits
                .into_iter()
                .map(|limit| limit.unwrap_or_default())
                .collect(),
        })
    }

    /// Explicit unlimited configuration for every physical domain.
    pub fn unlimited(topology: &MemoryTopology) -> Self {
        Self {
            topology: topology.identity,
            limits: vec![MemoryLimit::Unlimited; topology.len()],
        }
    }

    /// Re-expresses this authenticated vector using the topology's public names.
    /// The resulting declarations must still be resolved by their next backend.
    pub fn named(
        &self,
        topology: &MemoryTopology,
    ) -> Result<MemoryLimitDeclarations, MemoryDomainError> {
        self.validate(topology)?;
        Ok(MemoryLimitDeclarations::new(topology.domains().map(
            |(domain, description)| (description.name.clone(), self.limits[domain.slot]),
        )))
    }

    /// Host allocation envelope for constructing [`Self::named`], including
    /// overlap between the temporary tuple vector and final shared slice.
    pub fn named_initialization_bytes(
        &self,
        topology: &MemoryTopology,
    ) -> Result<u64, MemoryDomainError> {
        self.validate(topology)?;
        let rows = vector_bytes::<(String, MemoryLimit)>(topology.len())?;
        let mut bytes = rows
            .checked_mul(2)
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>() as u64))
            .ok_or(MemoryDomainError::Overflow)?;
        for (_, description) in topology.domains() {
            bytes = bytes
                .checked_add(
                    u64::try_from(description.name.len())
                        .map_err(|_| MemoryDomainError::Overflow)?,
                )
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }

    /// Rejects a complete vector belonging to another topology.
    pub fn validate(&self, topology: &MemoryTopology) -> Result<(), MemoryDomainError> {
        if self.topology != topology.identity || self.limits.len() != topology.len() {
            return Err(MemoryDomainError::ForeignTopology);
        }
        Ok(())
    }

    /// Limit for one validated domain.
    pub fn get(&self, domain: MemoryDomainId) -> Result<MemoryLimit, MemoryDomainError> {
        if domain.topology != self.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        self.limits
            .get(domain.slot)
            .copied()
            .ok_or(MemoryDomainError::UnknownDomain { domain })
    }

    /// Complete limits in topology slot order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (MemoryDomainId, MemoryLimit)> + '_ {
        self.limits
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, limit)| {
                (
                    MemoryDomainId {
                        topology: self.topology,
                        slot,
                    },
                    limit,
                )
            })
    }

    /// Pointwise intersection, with no aggregate request cap.
    pub fn intersection(&self, other: &Self) -> Result<Self, MemoryDomainError> {
        if self.topology != other.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        Ok(Self {
            topology: self.topology,
            limits: self
                .limits
                .iter()
                .zip(&other.limits)
                .map(|(&a, &b)| a.minimum(b))
                .collect(),
        })
    }

    /// Validates and raises a retained constraint pointwise without allocation.
    /// The caller must separately authenticate move-only succession authority.
    /// All identity checks precede mutation; values are copied in place.
    pub fn raise(&mut self, next: &Self) -> Result<(), MemoryDomainError> {
        if self.topology != next.topology || self.limits.len() != next.limits.len() {
            return Err(MemoryDomainError::ForeignTopology);
        }
        for (current, next) in self.limits.iter_mut().zip(&next.limits) {
            if current.is_raised_by(*next) {
                *current = *next;
            }
        }
        Ok(())
    }

    /// Removes every finite constraint without changing topology or allocation.
    /// This descriptor operation grants no authority to change a live account.
    pub fn make_unlimited(&mut self) {
        self.limits.fill(MemoryLimit::Unlimited);
    }

    /// Copies authenticated limits into an already allocated destination.
    pub fn copy_from(&mut self, source: &Self) -> Result<(), MemoryDomainError> {
        if self.topology != source.topology || self.limits.len() != source.limits.len() {
            return Err(MemoryDomainError::ForeignTopology);
        }
        self.limits.copy_from_slice(&source.limits);
        Ok(())
    }

    /// Controlled vector backing, excluding the owner's inline representation.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        vector_bytes::<MemoryLimit>(self.limits.capacity())
    }
}
