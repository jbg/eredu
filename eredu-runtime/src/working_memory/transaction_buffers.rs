//! Reusable, host-funded transaction storage and borrowed requirement projection.
use super::*;
mod reservation;

pub(in crate::working_memory) struct TransactionBuffers {
    charges: Vec<DomainMemoryCharge>,
    limits: MemoryLimits,
}
impl TransactionBuffers {
    pub(super) fn new(topology: &MemoryTopology) -> Self {
        Self {
            charges: vec![DomainMemoryCharge::default(); topology.len()],
            limits: MemoryLimits::unlimited(topology),
        }
    }
}

/// Borrows already attributed producer facts; normalization allocates nothing.
pub(in crate::working_memory) struct RequirementProjection<'a> {
    pub parts: &'a [&'a DomainMemoryRequirements],
    pub headroom: &'a eredu_core::MemoryHeadroomDeclarations,
    pub host_bytes: u64,
}
/// Closed runtime projections normalize into the coordinator's funded scratch.
/// Implementations borrow descriptions and perform no allocation or callbacks.
pub(in crate::working_memory) trait RequirementProjectionSource {
    fn fill(
        &self,
        topology: &MemoryTopology,
        charges: &mut [DomainMemoryCharge],
    ) -> Result<(), WorkingMemoryError>;
}
impl RequirementProjection<'_> {
    pub(in crate::working_memory) fn materialize(
        &self,
        topology: &MemoryTopology,
    ) -> Result<DomainMemoryRequirements, WorkingMemoryError> {
        let mut result = DomainMemoryRequirements::checked_sum(topology, self.parts)?;
        result.add_allocation(
            self.host_bytes,
            &eredu_core::MemoryPlacement::fixed(topology, topology.host_domain())?,
        )?;
        for (index, (name, bytes)) in self.headroom.iter().enumerate() {
            let domain = topology
                .domain_named(name)
                .ok_or(MemoryDomainError::UnknownDomainName { declaration: index })?;
            result.add_headroom(domain, bytes)?;
        }
        Ok(result)
    }
    pub(in crate::working_memory) fn backing_bytes(
        &self,
        topology: &MemoryTopology,
    ) -> Result<u64, WorkingMemoryError> {
        let dense = DomainMemoryRequirements::construction_backing_bytes(topology, 0)?;
        self.parts.iter().try_fold(dense, |sum, part| {
            part.validate(topology)?;
            sum.checked_add(
                part.clone_backing_bytes()?
                    .checked_sub(dense)
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
            .ok_or(WorkingMemoryError::Overflow)
        })
    }
}
impl RequirementProjectionSource for RequirementProjection<'_> {
    fn fill(
        &self,
        topology: &MemoryTopology,
        charges: &mut [DomainMemoryCharge],
    ) -> Result<(), WorkingMemoryError> {
        for part in self.parts {
            part.validate(topology)?;
        }
        for (index, (name, _)) in self.headroom.iter().enumerate() {
            topology
                .domain_named(name)
                .ok_or(MemoryDomainError::UnknownDomainName { declaration: index })?;
        }
        for (slot, (domain, description)) in topology.domains().enumerate() {
            let mut charge = DomainMemoryCharge::default();
            for part in self.parts {
                charge = charge.checked_add(part.get(domain)?)?;
            }
            if domain == topology.host_domain() {
                charge = charge.checked_add(DomainMemoryCharge {
                    accounted_bytes: self.host_bytes,
                    ..Default::default()
                })?;
            }
            for (_, bytes) in self
                .headroom
                .iter()
                .filter(|(name, _)| *name == description.name)
            {
                charge = charge.checked_add(DomainMemoryCharge {
                    headroom_bytes: bytes,
                    ..Default::default()
                })?;
            }
            charges[slot] = charge;
        }
        Ok(())
    }
}

impl MemoryLedger {
    /// One comparison funds account construction before its first allocation.
    /// The validator is a closed runtime source check, never a provider callback.
    pub(in crate::working_memory) fn accept_projected(
        &self,
        execution: &InferenceExecutionIdentity,
        projection: impl RequirementProjectionSource,
        declarations: &eredu_core::MemoryLimitDeclarations,
        ceiling: Option<&MemoryLimits>,
        handoffs: &[WorkingMemoryCapacityHandoff],
        control_floor: u64,
        validate: impl FnOnce(&Usage) -> Result<(), WorkingMemoryError>,
    ) -> Result<funding::PendingAccount, WorkingMemoryError> {
        // Buffers retire after the mutex loan if an internal check unwinds.
        let mut scratch;
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        while usage.pending_original.is_some() {
            usage = self
                .0
                .construction_ready
                .wait(usage)
                .map_err(|_| WorkingMemoryError::Poisoned)?;
        }
        scratch = usage
            .transaction_buffers
            .take()
            .ok_or(WorkingMemoryError::Poisoned)?;
        let result = (|| {
            projection.fill(self.topology(), &mut scratch.charges)?;
            scratch
                .limits
                .resolve_named_in_place(self.topology(), declarations, ceiling)?;
            validate(&usage)?;
            if control_floor > scratch.charges[usage.host_slot].total()? {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            usage
                .next_funding
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?;
            let commit = PreparedAccountCommit::prepare_requirement(
                self,
                execution,
                &usage,
                ReservationRequirement::Charges(&scratch.charges),
                Some(&scratch.limits),
                handoffs,
            )?;
            commit.commit(self, &mut usage);
            Ok(())
        })();
        if let Err(cause) = result {
            usage.transaction_buffers = Some(scratch);
            return Err(cause);
        }
        let pending = funding::PendingAccount::after_projected_commit(
            self,
            execution,
            &mut usage,
            scratch.charges,
            scratch.limits,
            control_floor,
        );
        drop(usage);
        // The accepted account now pays for construction and the replacement
        // scratch storage. Its pending ceilings remain visible throughout.
        let replacement = TransactionBuffers::new(self.topology());
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        usage.transaction_buffers = Some(replacement);
        drop(usage);
        Ok(pending)
    }
}
