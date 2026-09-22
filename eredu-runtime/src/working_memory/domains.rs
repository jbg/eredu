//! Dense physical-domain state owned by the single accounting coordinator.
use super::*;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct DomainUsage {
    pub(super) reserved: u64,
    pub(super) registered: u64,
    pub(super) registry_metadata: u64,
    pub(super) peak: u64,
    pub(super) placement_allowances: u64,
    pub(super) estimates: u64,
    pub(super) headroom: u64,
    pub(super) retiring_limit: MemoryLimit,
    pub(super) reset_retiring_limit: MemoryLimit,
}

#[derive(Debug)]
pub(super) struct FixedDomain {
    pub(super) existing: u64,
    pub(super) capacity: MemoryLimit,
}

// Host construction workers operate on the topology's host slot. Native
// publication and multi-domain transactions index explicit validated slots.
impl std::ops::Deref for Usage {
    type Target = DomainUsage;
    fn deref(&self) -> &Self::Target {
        &self.domains[self.host_slot]
    }
}
impl std::ops::DerefMut for Usage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.domains[self.host_slot]
    }
}
impl Usage {
    pub(super) fn domain_slot(&self, domain: MemoryDomainId) -> Result<usize, WorkingMemoryError> {
        self.domain_ids
            .iter()
            .position(|id| *id == domain)
            .ok_or(WorkingMemoryError::Domain(
                MemoryDomainError::ForeignTopology,
            ))
    }
}
impl std::ops::Deref for Pool {
    type Target = FixedDomain;
    fn deref(&self) -> &Self::Target {
        &self.domains[self.host_slot]
    }
}

impl Pool {
    pub(super) fn domain_capacity(
        &self,
        usage: &Usage,
        domain: MemoryDomainId,
        requested: Option<&MemoryLimits>,
    ) -> Result<MemoryLimit, WorkingMemoryError> {
        self.topology.slot(domain)?;
        let mut limit = self
            .limits
            .get(domain)?
            .minimum(usage.funding.capacity(domain)?);
        if usage.account_retiring.is_some() {
            limit = limit.minimum(usage.domains[self.topology.slot(domain)?].retiring_limit);
        }
        if let Some(account) = &usage.pending_original {
            limit = limit.minimum(account.capacity(domain)?);
        }
        limit = limit.minimum(super::resident_reset::capacity(usage, domain)?);
        if let Some(requested) = requested {
            limit = limit.minimum(requested.get(domain)?);
        }
        Ok(limit)
    }
}

/// Conservative estimation rule used for possible physical placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementAllowanceBasis {
    /// The complete backing capacity is reserved in every distinct physical
    /// domain declared by the allocation mechanism as a possible placement.
    /// Candidate locations sharing a physical domain receive one allowance.
    FullCapacityInEveryCandidateDomain,
}

/// One coherent physical-domain accounting observation, not measured residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDomainSnapshot {
    /// Physical domain authenticated by the ledger topology.
    pub domain: MemoryDomainId,
    /// Immutable coordinator limit for this physical domain.
    pub configured_limit: MemoryLimit,
    /// Minimum configured and applicable live account limits.
    pub effective_limit: MemoryLimit,
    /// Process-local fixed owners, including ledger controls in host memory.
    pub fixed_baseline: DomainMemoryCharge,
    /// Live registered backing charges, including placement allowances.
    pub registered_storage_bytes: u64,
    /// Host ownership and registry bookkeeping included in registered_storage_bytes.
    pub registry_metadata_bytes: u64,
    /// Unconverted account allowances retained by work and metadata owners.
    pub outstanding_reservation_bytes: u64,
    /// Retained reservation/account controls included in outstanding reservations.
    pub reservation_control_bytes: u64,
    /// Full-capacity candidate-domain allowances within the total charge.
    pub estimated_placement_allowance_bytes: u64,
    /// Conservative rule behind this domain's placement allowances. The
    /// allocation-specific mechanism and candidate set remain in its placement
    /// descriptor and the retained reservation requirements.
    pub placement_allowance_basis: Option<PlacementAllowanceBasis>,
    /// Admitted finite overhead upper estimates within the total charge.
    pub estimated_overhead_bytes: u64,
    /// Explicit domain headroom retained by applicable accounts.
    pub additional_headroom_bytes: u64,
    /// Checked baseline, registered and reserved charge in this domain.
    pub current_charge_bytes: u64,
    /// Greatest committed charge observed in this domain.
    pub historical_peak_bytes: u64,
}

/// Simultaneous ledger observations captured under its one coordinator lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLedgerSnapshot {
    /// Complete domain vector in immutable topology order.
    pub domains: Vec<MemoryDomainSnapshot>,
    /// Live reservation owners, including accepted original construction accounts.
    pub reservations: usize,
    /// Accounts that have entered allocation funding, retained by work, metadata
    /// or storage descendants. Accepted construction accounts that have not entered
    /// funding are counted only in `reservations`.
    pub funding_accounts: usize,
    /// Owners excluding admission because their storage is not fully quoted.
    pub unquoted_owners: usize,
}

impl MemoryLedger {
    /// Quotes a standalone reservation, including its retained report and
    /// account controls in host memory. `incremental` is a descriptive residual
    /// requirement; only admission with authenticated source pins may use that
    /// credit. This observation reserves nothing and grants no execution right.
    /// Preparations with a separately funded report owner use that owner's
    /// quotation instead of adding this standalone metadata charge again.
    pub fn reservation_requirements(
        &self,
        admission: &Admission,
        incremental: Option<&DomainMemoryRequirements>,
    ) -> Result<DomainMemoryRequirements, WorkingMemoryError> {
        if !eredu_core::AdmissionStateRequirements::from(&admission.state).physical_domains {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let state = admission
            .state
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let workspace = admission
            .state
            .execution_workspace
            .as_ref()
            .and_then(|value| value.physical_domains.as_ref())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if state.geometry != workspace.geometry {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut requirements = match incremental {
            Some(value) => value.clone(),
            None => state
                .requirements()?
                .checked_add(&workspace.requirements()?)?,
        };
        requirements.validate(self.topology())?;
        requirements =
            requirements.checked_add(&admission.additional_headroom.resolve(self.topology())?)?;
        let limits = admission.memory_limits.resolve(self.topology())?;
        let retained = admission.clone();
        self.add_standalone_controls(&retained, &mut requirements, &limits)?;
        Ok(requirements)
    }

    pub(super) fn add_standalone_controls(
        &self,
        retained: &Admission,
        requirements: &mut DomainMemoryRequirements,
        limits: &MemoryLimits,
    ) -> Result<u64, WorkingMemoryError> {
        let bytes = u64::try_from(reservation_metadata::constructor_bytes(
            self.topology(),
            requirements,
            limits,
        )?)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .checked_add(reservation_metadata::admission_backing_bytes(retained)?)
        .ok_or(WorkingMemoryError::Overflow)?;
        requirements.add_allocation(bytes, &self.0.host_placement)?;
        Ok(bytes)
    }

    /// Backend-established immutable physical sharing and location mapping.
    pub fn topology(&self) -> &MemoryTopology {
        &self.0.topology
    }

    /// Immutable process-local physical-domain limits established at construction.
    pub fn configured_limits(&self) -> &MemoryLimits {
        &self.0.limits
    }

    /// Retains the ledger-owned host placement descriptor without allocating.
    pub fn host_placement_handle(&self) -> Arc<eredu_core::MemoryPlacement> {
        Arc::clone(&self.0.host_placement)
    }
    /// Borrow the immutable placement of host metadata and storage.
    pub fn host_placement(&self) -> &eredu_core::MemoryPlacement {
        &self.0.host_placement
    }

    /// Retains the same topology identity without reconstructing its domains.
    pub fn topology_handle(&self) -> Arc<MemoryTopology> {
        Arc::clone(&self.0.topology)
    }

    /// Captures every domain under one lock. Allocation occurs before locking.
    /// The returned diagnostic container belongs to the caller, outside the
    /// admitted execution's retained storage. Observation reserves no capacity
    /// and grants no allocation, source, or execution authority.
    pub fn snapshot(&self) -> Result<MemoryLedgerSnapshot, WorkingMemoryError> {
        let mut domains = Vec::with_capacity(self.topology().len());
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        for (slot, (domain, _)) in self.topology().domains().enumerate() {
            let current = &usage.domains[slot];
            let baseline = self.0.baseline.get(domain)?;
            let charge = baseline
                .total()?
                .checked_add(current.registered)
                .and_then(|n| n.checked_add(current.reserved))
                .ok_or(WorkingMemoryError::Overflow)?;
            domains.push(MemoryDomainSnapshot {
                domain,
                configured_limit: self.0.limits.get(domain)?,
                effective_limit: self.0.domain_capacity(&usage, domain, None)?,
                fixed_baseline: baseline,
                registered_storage_bytes: current.registered,
                registry_metadata_bytes: current.registry_metadata,
                outstanding_reservation_bytes: current.reserved,
                reservation_control_bytes: if slot == self.0.host_slot {
                    usage.funding.control_bytes()?
                } else {
                    0
                },
                estimated_placement_allowance_bytes: current
                    .placement_allowances
                    .checked_add(baseline.placement_allowance_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?,
                placement_allowance_basis: (current.placement_allowances != 0
                    || baseline.placement_allowance_bytes != 0)
                    .then_some(PlacementAllowanceBasis::FullCapacityInEveryCandidateDomain),
                estimated_overhead_bytes: current
                    .estimates
                    .checked_add(baseline.estimated_overhead_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?,
                additional_headroom_bytes: current
                    .headroom
                    .checked_add(baseline.headroom_bytes)
                    .ok_or(WorkingMemoryError::Overflow)?,
                current_charge_bytes: charge,
                historical_peak_bytes: current.peak,
            });
        }
        let snapshot = MemoryLedgerSnapshot {
            domains,
            reservations: usage.reservations,
            funding_accounts: usage.funding.len(),
            unquoted_owners: usage.unquoted_owners,
        };
        Ok(snapshot)
    }
}
