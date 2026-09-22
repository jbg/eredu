use super::*;

/// Distinct accounting categories in one physical domain.
///
/// None of these counters is a measurement of process or device residency.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DomainMemoryCharge {
    /// Allocation capacities with known fixed physical placement.
    pub accounted_bytes: u64,
    /// Conservative full-capacity allowances for possible placement.
    pub placement_allowance_bytes: u64,
    /// Upper endpoints of finite overhead estimates.
    pub estimated_overhead_bytes: u64,
    /// Explicit additional allowance selected by the application.
    pub headroom_bytes: u64,
}

impl DomainMemoryCharge {
    /// Complete checked charge. Unlimited limits do not bypass this operation.
    pub fn total(self) -> Result<u64, MemoryDomainError> {
        self.accounted_bytes
            .checked_add(self.placement_allowance_bytes)
            .and_then(|bytes| bytes.checked_add(self.estimated_overhead_bytes))
            .and_then(|bytes| bytes.checked_add(self.headroom_bytes))
            .ok_or(MemoryDomainError::Overflow)
    }

    /// Adds simultaneous contributions without deduplicating diagnostic labels.
    pub fn checked_add(self, other: Self) -> Result<Self, MemoryDomainError> {
        let result = Self {
            accounted_bytes: self
                .accounted_bytes
                .checked_add(other.accounted_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            placement_allowance_bytes: self
                .placement_allowance_bytes
                .checked_add(other.placement_allowance_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            estimated_overhead_bytes: self
                .estimated_overhead_bytes
                .checked_add(other.estimated_overhead_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            headroom_bytes: self
                .headroom_bytes
                .checked_add(other.headroom_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
        };
        result.total()?;
        Ok(result)
    }

    /// Removes a known contribution. Underflow is an error, never saturation.
    pub fn checked_sub(self, other: Self) -> Result<Self, MemoryDomainError> {
        self.total()?;
        other.total()?;
        Ok(Self {
            accounted_bytes: self
                .accounted_bytes
                .checked_sub(other.accounted_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            placement_allowance_bytes: self
                .placement_allowance_bytes
                .checked_sub(other.placement_allowance_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            estimated_overhead_bytes: self
                .estimated_overhead_bytes
                .checked_sub(other.estimated_overhead_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
            headroom_bytes: self
                .headroom_bytes
                .checked_sub(other.headroom_bytes)
                .ok_or(MemoryDomainError::Overflow)?,
        })
    }
}

/// One conservative placement contribution, retaining its identified basis.
/// Repeated equal descriptors are independent contributions, never storage keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementAllowance {
    /// Full capacity of one backing allocation or prospective allocation bound.
    pub backing_bytes: u64,
    /// Candidate domains and backend-supplied derivation.
    pub placement: MemoryPlacement,
}

/// Finite estimated overhead retained with its source, range, and derivation.
/// Its upper endpoint is an allowance, not a guaranteed allocation bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainOverheadEstimate {
    /// Physical domain to which the upper estimated endpoint contributes.
    pub domain: MemoryDomainId,
    /// Diagnostic source, never a storage identity or a reason to deduplicate.
    pub source: String,
    /// Validated inclusive estimated range.
    pub range: crate::FiniteMemoryEstimate,
    /// Derivation supporting the finite estimate.
    pub basis: String,
}

/// Checked simultaneous requirements resolved before accumulation to physical
/// domains. This descriptor provides no storage credit or execution authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainMemoryRequirements {
    topology: u64,
    charges: Vec<DomainMemoryCharge>,
    placement_allowances: Vec<PlacementAllowance>,
    overhead_estimates: Vec<DomainOverheadEstimate>,
}

impl DomainMemoryRequirements {
    /// Composes attributed producers into one dense vector, preserving all
    /// placement and overhead provenance. Callers fund the result before
    /// construction; no intermediate owned requirement vectors are created.
    pub fn checked_sum(
        topology: &MemoryTopology,
        parts: &[&Self],
    ) -> Result<Self, MemoryDomainError> {
        let mut allowances = 0usize;
        let mut estimates = 0usize;
        for part in parts {
            part.validate(topology)?;
            allowances = allowances
                .checked_add(part.placement_allowances.len())
                .ok_or(MemoryDomainError::Overflow)?;
            estimates = estimates
                .checked_add(part.overhead_estimates.len())
                .ok_or(MemoryDomainError::Overflow)?;
        }
        // Validate all arithmetic before constructing any result storage.
        for (domain, _) in topology.domains() {
            parts
                .iter()
                .try_fold(DomainMemoryCharge::default(), |sum, part| {
                    sum.checked_add(part.get(domain)?)
                })?;
        }
        let mut result = Self {
            topology: topology.identity,
            charges: vec![DomainMemoryCharge::default(); topology.len()],
            placement_allowances: Vec::with_capacity(allowances),
            overhead_estimates: Vec::with_capacity(estimates),
        };
        for part in parts {
            for (out, value) in result.charges.iter_mut().zip(&part.charges) {
                *out = out.checked_add(*value).expect("validated sum");
            }
            result
                .placement_allowances
                .extend_from_slice(&part.placement_allowances);
            result
                .overhead_estimates
                .extend_from_slice(&part.overhead_estimates);
        }
        Ok(result)
    }

    /// An explicit empty requirement with the same immutable topology identity.
    pub fn empty_like(&self) -> Self {
        Self {
            topology: self.topology,
            charges: vec![DomainMemoryCharge::default(); self.charges.len()],
            placement_allowances: Vec::new(),
            overhead_estimates: Vec::new(),
        }
    }

    /// Per-domain peak across safely separated completed spans, retaining the
    /// categories of the highest charged span in each domain. Descriptions
    /// retain the alternative spans' estimation bases;
    /// they are explanatory provenance and are not added to these peak counters.
    pub fn checked_peak(&self, other: &Self) -> Result<Self, MemoryDomainError> {
        if self.topology != other.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        let charges = self
            .charges
            .iter()
            .zip(&other.charges)
            .map(|(left, right)| {
                Ok(if left.total()? >= right.total()? {
                    *left
                } else {
                    *right
                })
            })
            .collect::<Result<Vec<_>, MemoryDomainError>>()?;
        self.placement_allowances
            .len()
            .checked_add(other.placement_allowances.len())
            .ok_or(MemoryDomainError::Overflow)?;
        self.overhead_estimates
            .len()
            .checked_add(other.overhead_estimates.len())
            .ok_or(MemoryDomainError::Overflow)?;
        let mut placement_allowances =
            Vec::with_capacity(self.placement_allowances.len() + other.placement_allowances.len());
        placement_allowances.extend_from_slice(&self.placement_allowances);
        placement_allowances.extend_from_slice(&other.placement_allowances);
        let mut overhead_estimates =
            Vec::with_capacity(self.overhead_estimates.len() + other.overhead_estimates.len());
        overhead_estimates.extend_from_slice(&self.overhead_estimates);
        overhead_estimates.extend_from_slice(&other.overhead_estimates);
        Ok(Self {
            topology: self.topology,
            charges,
            placement_allowances,
            overhead_estimates,
        })
    }
    /// Constructs empty dense counters and an exact prospective number of
    /// candidate-placement descriptions. Callers fund these host allocations
    /// before construction; subsequent pushes up to this population do not grow.
    pub fn zero_with_allowance_capacity(topology: &MemoryTopology, allowances: usize) -> Self {
        Self {
            topology: topology.identity,
            charges: vec![DomainMemoryCharge::default(); topology.len()],
            placement_allowances: Vec::with_capacity(allowances),
            overhead_estimates: Vec::new(),
        }
    }

    /// Requested dense counter and allowance-cell backing, excluding separately
    /// copied candidate lists and estimation text and the inline owner shell.
    pub fn construction_backing_bytes(
        topology: &MemoryTopology,
        allowances: usize,
    ) -> Result<u64, MemoryDomainError> {
        vector_bytes::<DomainMemoryCharge>(topology.len())?
            .checked_add(vector_bytes::<PlacementAllowance>(allowances)?)
            .ok_or(MemoryDomainError::Overflow)
    }
    /// An explicit zero contribution in every domain of this topology.
    pub fn zero(topology: &MemoryTopology) -> Self {
        Self {
            topology: topology.identity,
            charges: vec![DomainMemoryCharge::default(); topology.len()],
            placement_allowances: Vec::new(),
            overhead_estimates: Vec::new(),
        }
    }

    /// Verifies the exact topology identity and complete dense vector.
    pub fn validate(&self, topology: &MemoryTopology) -> Result<(), MemoryDomainError> {
        if self.topology != topology.identity || self.charges.len() != topology.len() {
            return Err(MemoryDomainError::ForeignTopology);
        }
        Ok(())
    }

    fn slot(&self, domain: MemoryDomainId) -> Result<usize, MemoryDomainError> {
        if domain.topology != self.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        if domain.slot >= self.charges.len() {
            return Err(MemoryDomainError::UnknownDomain { domain });
        }
        Ok(domain.slot)
    }

    /// Per-domain categories; no sum of placement allowances is presented as
    /// actual physical residency.
    pub fn get(&self, domain: MemoryDomainId) -> Result<DomainMemoryCharge, MemoryDomainError> {
        Ok(self.charges[self.slot(domain)?])
    }

    /// All domain requirements in immutable topology order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (MemoryDomainId, DomainMemoryCharge)> + '_ {
        self.charges
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, charge)| {
                (
                    MemoryDomainId {
                        topology: self.topology,
                        slot,
                    },
                    charge,
                )
            })
    }

    /// Identified candidate-placement estimates, including zero-byte declarations.
    pub fn placement_allowances(&self) -> &[PlacementAllowance] {
        &self.placement_allowances
    }

    /// All identified finite overhead estimates, including their lower endpoints.
    pub fn overhead_estimates(&self) -> &[DomainOverheadEstimate] {
        &self.overhead_estimates
    }

    /// Adds a finite estimated contribution. Repeated diagnostic labels remain
    /// independent charges. This does not apply inference overhead policy.
    pub fn add_estimate(
        &mut self,
        estimate: DomainOverheadEstimate,
    ) -> Result<(), MemoryDomainError> {
        let slot = self.slot(estimate.domain)?;
        if estimate.source.trim().is_empty() || estimate.basis.trim().is_empty() {
            return Err(MemoryDomainError::InvalidEstimateDescription);
        }
        let charge = self.charges[slot].checked_add(DomainMemoryCharge {
            estimated_overhead_bytes: estimate.range.upper_bytes(),
            ..Default::default()
        })?;
        self.overhead_estimates.push(estimate);
        self.charges[slot] = charge;
        Ok(())
    }

    /// Adds one independent backing allocation. Aliases must be deduplicated by
    /// authenticated storage identity before calling this operation. All domains
    /// are validated before any mutation; rejection leaves the descriptor intact.
    pub fn add_allocation(
        &mut self,
        bytes: u64,
        placement: &MemoryPlacement,
    ) -> Result<(), MemoryDomainError> {
        let charge = match placement.kind() {
            MemoryPlacementKind::Fixed(_) => DomainMemoryCharge {
                accounted_bytes: bytes,
                ..Default::default()
            },
            MemoryPlacementKind::Possible { .. } => DomainMemoryCharge {
                placement_allowance_bytes: bytes,
                ..Default::default()
            },
        };
        for &domain in placement.domains() {
            self.charges[self.slot(domain)?].checked_add(charge)?;
        }
        // Prepare descriptor storage before committing any counter. Runtime
        // admission must fund this construction separately in the host domain.
        if matches!(placement.kind(), MemoryPlacementKind::Possible { .. }) {
            self.placement_allowances.push(PlacementAllowance {
                backing_bytes: bytes,
                placement: placement.clone(),
            });
        }
        for &domain in placement.domains() {
            self.charges[domain.slot] = self.charges[domain.slot]
                .checked_add(charge)
                .expect("validated domain addition");
        }
        Ok(())
    }

    /// Adds application headroom to a specified physical domain. Headroom does
    /// not establish finite bounds for unknown dependency overhead.
    pub fn add_headroom(
        &mut self,
        domain: MemoryDomainId,
        bytes: u64,
    ) -> Result<(), MemoryDomainError> {
        let slot = self.slot(domain)?;
        let next = self.charges[slot].checked_add(DomainMemoryCharge {
            headroom_bytes: bytes,
            ..Default::default()
        })?;
        self.charges[slot] = next;
        Ok(())
    }

    /// Removes a separately identified fixed allocation contribution while
    /// preserving candidate-placement and estimated-overhead provenance.
    /// This changes a descriptive requirement only; it never releases storage.
    pub fn subtract_accounted(
        &mut self,
        domain: MemoryDomainId,
        bytes: u64,
    ) -> Result<(), MemoryDomainError> {
        let slot = self.slot(domain)?;
        self.charges[slot] = self.charges[slot].checked_sub(DomainMemoryCharge {
            accounted_bytes: bytes,
            ..Default::default()
        })?;
        Ok(())
    }

    /// Removes an explicitly retained component of a descriptive composition.
    /// Equal provenance entries retain their multiplicity: removing one producer
    /// cannot erase another producer with the same description. Every domain and
    /// provenance entry is validated before changing this report; rejection leaves
    /// it unchanged. This allocates nothing and grants no storage credit or authority.
    pub fn subtract_component(&mut self, component: &Self) -> Result<(), MemoryDomainError> {
        if self.topology != component.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        if self.charges.len() != component.charges.len() {
            return Err(MemoryDomainError::MissingRequirementComponent);
        }
        for (whole, part) in self.charges.iter().zip(&component.charges) {
            whole.checked_sub(*part)?;
        }
        fn contains_all<T: PartialEq>(whole: &[T], part: &[T]) -> bool {
            part.iter().enumerate().all(|(index, value)| {
                whole.iter().filter(|candidate| *candidate == value).count()
                    >= part[..=index]
                        .iter()
                        .filter(|candidate| *candidate == value)
                        .count()
            })
        }
        if !contains_all(&self.placement_allowances, &component.placement_allowances)
            || !contains_all(&self.overhead_estimates, &component.overhead_estimates)
        {
            return Err(MemoryDomainError::MissingRequirementComponent);
        }
        for (whole, part) in self.charges.iter_mut().zip(&component.charges) {
            *whole = whole
                .checked_sub(*part)
                .expect("validated component subtraction");
        }
        fn remove_all<T: PartialEq>(whole: &mut Vec<T>, part: &[T]) {
            for value in part {
                let index = whole
                    .iter()
                    .position(|candidate| candidate == value)
                    .expect("validated component provenance");
                whole.remove(index);
            }
        }
        remove_all(
            &mut self.placement_allowances,
            &component.placement_allowances,
        );
        remove_all(&mut self.overhead_estimates, &component.overhead_estimates);
        Ok(())
    }

    /// Combines simultaneously live requirements, including independent
    /// allocations that have equal capacities and placement descriptions.
    pub fn checked_add(&self, other: &Self) -> Result<Self, MemoryDomainError> {
        if self.topology != other.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        let charges = self
            .charges
            .iter()
            .zip(&other.charges)
            .map(|(&a, &b)| a.checked_add(b))
            .collect::<Result<Vec<_>, _>>()?;
        self.placement_allowances
            .len()
            .checked_add(other.placement_allowances.len())
            .ok_or(MemoryDomainError::Overflow)?;
        self.overhead_estimates
            .len()
            .checked_add(other.overhead_estimates.len())
            .ok_or(MemoryDomainError::Overflow)?;
        let mut placement_allowances =
            Vec::with_capacity(self.placement_allowances.len() + other.placement_allowances.len());
        placement_allowances.extend_from_slice(&self.placement_allowances);
        placement_allowances.extend_from_slice(&other.placement_allowances);
        let mut overhead_estimates =
            Vec::with_capacity(self.overhead_estimates.len() + other.overhead_estimates.len());
        overhead_estimates.extend_from_slice(&self.overhead_estimates);
        overhead_estimates.extend_from_slice(&other.overhead_estimates);
        Ok(Self {
            topology: self.topology,
            charges,
            placement_allowances,
            overhead_estimates,
        })
    }

    /// Pure capacity comparison against the complete existing live charge.
    /// Runtime must perform this comparison under its coordinator lock and
    /// publish every domain atomically; this method reserves nothing.
    pub fn check_increment(
        &self,
        existing: &Self,
        limits: &MemoryLimits,
    ) -> Result<(), MemoryDomainError> {
        if self.topology != existing.topology {
            return Err(MemoryDomainError::ForeignTopology);
        }
        for (domain, charge) in self.iter() {
            limits
                .get(domain)?
                .check(domain, existing.get(domain)?.total()?, charge.total()?)?;
        }
        Ok(())
    }

    /// Controlled vector and descriptor backing to include in host metadata
    /// funding. Inline owner representations are priced by their containing owner.
    pub fn backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        let mut bytes = vector_bytes::<DomainMemoryCharge>(self.charges.capacity())?
            .checked_add(vector_bytes::<PlacementAllowance>(
                self.placement_allowances.capacity(),
            )?)
            .and_then(|bytes| {
                bytes.checked_add(
                    vector_bytes::<DomainOverheadEstimate>(self.overhead_estimates.capacity())
                        .ok()?,
                )
            })
            .ok_or(MemoryDomainError::Overflow)?;
        for allowance in &self.placement_allowances {
            bytes = bytes
                .checked_add(allowance.placement.backing_bytes()?)
                .ok_or(MemoryDomainError::Overflow)?;
        }
        for estimate in &self.overhead_estimates {
            bytes = bytes
                .checked_add(vector_bytes::<u8>(estimate.source.capacity())?)
                .and_then(|bytes| {
                    bytes.checked_add(vector_bytes::<u8>(estimate.basis.capacity()).ok()?)
                })
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }

    /// Prospective backing for an owned deep copy, including all provenance.
    /// This allocates nothing and does not charge unused source capacity again.
    pub fn clone_backing_bytes(&self) -> Result<u64, MemoryDomainError> {
        let mut bytes = vector_bytes::<DomainMemoryCharge>(self.charges.len())?
            .checked_add(vector_bytes::<PlacementAllowance>(
                self.placement_allowances.len(),
            )?)
            .and_then(|n| {
                n.checked_add(
                    vector_bytes::<DomainOverheadEstimate>(self.overhead_estimates.len()).ok()?,
                )
            })
            .ok_or(MemoryDomainError::Overflow)?;
        for allowance in &self.placement_allowances {
            bytes = bytes
                .checked_add(allowance.placement.clone_backing_bytes()?)
                .ok_or(MemoryDomainError::Overflow)?;
        }
        for estimate in &self.overhead_estimates {
            bytes = bytes
                .checked_add(vector_bytes::<u8>(estimate.source.len())?)
                .and_then(|n| n.checked_add(vector_bytes::<u8>(estimate.basis.len()).ok()?))
                .ok_or(MemoryDomainError::Overflow)?;
        }
        Ok(bytes)
    }
}
