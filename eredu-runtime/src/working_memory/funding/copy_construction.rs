//! Copy account construction starts only after the shared domain transaction.
use super::copy_controls::CopyAccountState;
use super::*;
use crate::working_memory::transaction_buffers::RequirementProjection;

pub(in crate::working_memory) struct PreparedCopyAccount {
    // Descriptor storage retires before its construction ticket on failure.
    requirements: eredu_core::DomainMemoryRequirements,
    ticket: AccountTicket,
    state: CopyAccountState,
}
impl PreparedCopyAccount {
    pub(in crate::working_memory) fn accept(
        pool: &MemoryLedger,
        source_execution: &InferenceExecutionIdentity,
        mut projection: RequirementProjection<'_>,
        declarations: &eredu_core::MemoryLimitDeclarations,
        controls: u64,
        holds: CopyHostHolds,
        validate: impl FnOnce(&Usage) -> Result<(), WorkingMemoryError>,
    ) -> Result<Self, WorkingMemoryError> {
        projection.host_bytes = projection
            .host_bytes
            .checked_add(controls)
            .ok_or(WorkingMemoryError::Overflow)?;
        let host_domain = pool.topology().host_domain();
        let host_total = projection
            .parts
            .iter()
            .try_fold(projection.host_bytes, |sum, part| {
                sum.checked_add(part.get(host_domain)?.total()?)
                    .ok_or(WorkingMemoryError::Overflow)
            })?;
        let state = CopyAccountState::prepare(controls, holds, host_total)?;
        let pending = pool.accept_projected(
            source_execution,
            RequirementProjection {
                parts: projection.parts,
                headroom: projection.headroom,
                host_bytes: projection.host_bytes,
            },
            declarations,
            None,
            &[],
            controls,
            validate,
        )?;
        let ticket = pending.publish();
        ticket.status()?;
        let requirements = projection.materialize(pool.topology())?;
        Ok(Self {
            requirements,
            ticket,
            state,
        })
    }
    pub(super) fn finish(
        self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(MemoryLedger, u64, eredu_core::DomainMemoryRequirements), WorkingMemoryError> {
        let Self {
            requirements,
            ticket,
            state,
        } = self;
        let pool = ticket.pool().clone();
        let id = ticket.into_copy(state, execution)?;
        Ok((pool, id, requirements))
    }
    pub(in crate::working_memory) fn workspace(
        self,
        execution: &InferenceExecutionIdentity,
        pin: Option<RegisteredStoragePin>,
    ) -> Result<
        (
            eredu_core::DomainMemoryRequirements,
            WorkingMemoryFundingRun,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        let (pool, id, requirements) = self.finish(execution)?;
        Ok((
            requirements,
            run(&pool, id),
            scope(&pool, id, ScopePurpose::Native, pin),
        ))
    }
    pub(in crate::working_memory) fn sampling(
        self,
        execution: &InferenceExecutionIdentity,
        pin: RegisteredStoragePin,
        sampler: u64,
    ) -> Result<
        (
            eredu_core::DomainMemoryRequirements,
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        let (pool, id, requirements) = self.finish(execution)?;
        Ok((
            requirements,
            run(&pool, id),
            WorkingMemorySamplerScope {
                scope: Some(scope(&pool, id, ScopePurpose::Host, None)),
                held: sampler,
            },
            scope(&pool, id, ScopePurpose::Native, Some(pin)),
        ))
    }
    pub(in crate::working_memory) fn paired(
        self,
        execution: &InferenceExecutionIdentity,
        pin: RegisteredStoragePin,
        sampler: u64,
        decoder: u64,
    ) -> Result<
        (
            eredu_core::DomainMemoryRequirements,
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            WorkingMemoryDecoderHostScope,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        let (pool, id, requirements) = self.finish(execution)?;
        Ok((
            requirements,
            run(&pool, id),
            WorkingMemorySamplerScope {
                scope: Some(scope(&pool, id, ScopePurpose::Host, None)),
                held: sampler,
            },
            decoder_scope(&pool, id, decoder),
            scope(&pool, id, ScopePurpose::Native, Some(pin)),
        ))
    }
    pub(in crate::working_memory) fn grouped(
        self,
        execution: &InferenceExecutionIdentity,
        pin: RegisteredStoragePin,
        sampler: u64,
        holds: &[u64],
    ) -> Result<
        (
            eredu_core::DomainMemoryRequirements,
            WorkingMemoryFundingRun,
            WorkingMemorySamplerScope,
            Vec<WorkingMemoryDecoderHostScope>,
            WorkingMemoryFundingScope,
        ),
        WorkingMemoryError,
    > {
        let mut scopes = crate::working_memory::qualified_storage::vector(holds.len(), true)?;
        let (pool, id, requirements) = self.finish(execution)?;
        for &held in holds {
            scopes.push(decoder_scope(&pool, id, held));
        }
        Ok((
            requirements,
            run(&pool, id),
            WorkingMemorySamplerScope {
                scope: Some(scope(&pool, id, ScopePurpose::Host, None)),
                held: sampler,
            },
            scopes,
            scope(&pool, id, ScopePurpose::Native, Some(pin)),
        ))
    }
    pub(super) fn host(
        self,
        execution: &InferenceExecutionIdentity,
        held: u64,
    ) -> Result<WorkingMemoryDecoderHostScope, WorkingMemoryError> {
        let (pool, id, requirements) = self.finish(execution)?;
        drop(requirements);
        Ok(decoder_scope(&pool, id, held))
    }
}
fn run(pool: &MemoryLedger, id: u64) -> WorkingMemoryFundingRun {
    WorkingMemoryFundingRun {
        pool: pool.clone(),
        id,
        open: true,
        handoff_taken: false,
        borrowed_storage: None,
    }
}
fn scope(
    pool: &MemoryLedger,
    id: u64,
    purpose: ScopePurpose,
    borrowed_storage: Option<RegisteredStoragePin>,
) -> WorkingMemoryFundingScope {
    WorkingMemoryFundingScope {
        purpose,
        pool: pool.clone(),
        id,
        active: true,
        borrowed_storage,
        capture_source: None,
        native_publication_identity: None,
        allocation_funding: None,
    }
}
fn decoder_scope(pool: &MemoryLedger, id: u64, held: u64) -> WorkingMemoryDecoderHostScope {
    WorkingMemoryDecoderHostScope {
        scope: Some(scope(pool, id, ScopePurpose::Host, None)),
        held,
    }
}

/// Dense transaction buffers are moved into the accepted account. The baseline
/// pays for their replacement; the account pays for the moved vectors and its
/// retained domain report. Shared source preparation pays its quoted fixed layout.
pub(in crate::working_memory) fn copy_domain_controls(
    pool: &MemoryLedger,
    projection: &RequirementProjection<'_>,
) -> Result<u64, WorkingMemoryError> {
    let domain_vectors = pool
        .topology()
        .len()
        .checked_mul(
            std::mem::size_of::<eredu_core::MemoryLimit>()
                + std::mem::size_of::<eredu_core::DomainMemoryCharge>(),
        )
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    projection
        .backing_bytes(pool.topology())?
        .checked_add(domain_vectors)
        .and_then(|n| n.checked_add(domain_balance_bytes(pool.topology()).ok()?))
        .and_then(|n| {
            n.checked_add(
                u64::try_from(
                    std::mem::size_of::<PreparedCopyAccount>()
                        + std::mem::size_of::<PendingAccount>()
                        + std::mem::size_of::<RequirementProjection<'_>>(),
                )
                .ok()?,
            )
        })
        .ok_or(WorkingMemoryError::Overflow)
}
