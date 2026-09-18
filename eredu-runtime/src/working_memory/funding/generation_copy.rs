//! Independent finite provider destinations, using the existing pending account.
use super::*;
use crate::working_memory::OriginalTextControlGuard;
use std::{alloc::Layout, mem::size_of, sync::atomic::AtomicUsize};

pub(in crate::working_memory) enum GenerationCopySource<'a> {
    Original(&'a OriginalTextControlGuard, &'a WorkingMemoryReservation),
    Copied(&'a GenerationCopyCustody),
}
impl GenerationCopySource<'_> {
    fn pool(&self) -> &WorkingMemoryPool {
        match self {
            Self::Original(c, _) => c.custody.raw().pool(),
            Self::Copied(c) => c.inner().ticket.pool(),
        }
    }
    pub(in crate::working_memory) fn execution(&self) -> &InferenceExecutionIdentity {
        match self {
            Self::Original(_, r) => &r.0.execution,
            Self::Copied(c) => &c.inner().execution,
        }
    }
    fn validate(&self, usage: &Usage) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Original(c, r) => c.custody.validate_issuance_locked(self.pool(), usage, r),
            Self::Copied(c) => usage.funding.validate_constructing(c.inner().ticket.id()),
        }
    }
}
#[derive(Debug)]
struct Account {
    execution: InferenceExecutionIdentity,
    // Scalar logical budget only: this lease never retains H or this account.
    snapshot: Option<crate::execution_control::SnapshotReservation>,
    ticket: AccountTicket,
}
/// No native scope or request/source authority exists in this shared host owner.
#[derive(Debug)]
pub(in crate::working_memory) struct GenerationCopyCustody(Option<Arc<Account>>);
impl Clone for GenerationCopyCustody {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live copy account"),
        )))
    }
}
impl Drop for GenerationCopyCustody {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(in crate::working_memory) struct GenerationCopyAdmission {
    #[source]
    pub(in crate::working_memory) cause: WorkingMemoryError,
    pub(in crate::working_memory) partial: Option<GenerationCopyCustody>,
}
impl GenerationCopyCustody {
    fn inner(&self) -> &Account {
        self.0.as_deref().expect("live copy account")
    }
    // Called immediately after admission, before any provider or authority alias.
    pub(in crate::working_memory) fn attach_snapshot_resume(
        &mut self,
        reservation: crate::execution_control::PendingSnapshotResumeRetention,
    ) {
        let account = Arc::get_mut(self.0.as_mut().expect("live copy account"))
            .expect("new host copy has no published aliases");
        assert!(
            account.snapshot.is_none(),
            "one resume lease per copied provider"
        );
        account.snapshot = Some(reservation.publish());
    }
    pub(in crate::working_memory) fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.inner().ticket.status()
    }
    pub(in crate::working_memory) fn pool(&self) -> &WorkingMemoryPool {
        self.inner().ticket.pool()
    }
    pub(in crate::working_memory) fn control_bytes() -> Option<usize> {
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Account>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            eredu_core::HostPreparationAuthority::retention_bytes::<Self>()?,
            arc,
            size_of::<Account>(),
            size_of::<Arc<Account>>(),
            size_of::<Option<Arc<Account>>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<Account>>(),
            size_of::<Self>(),
            size_of::<AccountNode>(),
            size_of::<PendingOriginal>(),
            size_of::<PendingAccount>(),
            size_of::<AccountTicket>(),
            size_of::<PreparedAccountCommit<'_>>(),
            size_of::<GenerationCopySource<'_>>(),
            size_of::<GenerationCopyAdmission>(),
            size_of::<Result<Self, GenerationCopyAdmission>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    // The only caller is the concrete retained generation copier. Its source-
    // derived complete requirement is never accepted from a public byte caller.
    pub(in crate::working_memory) fn admit(
        source: GenerationCopySource<'_>,
        bytes: u64,
        capacity: u64,
    ) -> Result<Self, GenerationCopyAdmission> {
        let rejected = |cause| GenerationCopyAdmission {
            cause,
            partial: None,
        };
        let pool = source.pool();
        let execution = source.execution();
        let pending = {
            let mut usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| rejected(WorkingMemoryError::Poisoned))?;
            source.validate(&usage).map_err(rejected)?;
            let commit =
                PreparedAccountCommit::prepare(pool, execution, &usage, bytes, Some(capacity), &[])
                    .map_err(rejected)?;
            PendingAccount::accept(
                pool,
                execution,
                &mut usage,
                commit,
                bytes,
                Some(capacity),
                bytes,
            )
            .map_err(rejected)?
        };
        // Every allocation below follows the accepted pending account. Its node
        // and closed Arc deallocate before the ticket finally refunds the hold.
        let ticket = pending.publish();
        let out = Self(Some(Arc::new(Account {
            execution: execution.clone(),
            snapshot: None,
            ticket,
        })));
        match out.validate() {
            Ok(()) => Ok(out),
            Err(cause) => Err(GenerationCopyAdmission {
                cause,
                partial: Some(out),
            }),
        }
    }
}
