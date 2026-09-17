//! Existing original source publication over a completed GPU-to-Host copy.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    NativeStorageRegistration, OriginalHostSourceBank, OriginalHostSourceConstruction,
    OriginalHostSourceCustody, OriginalHostSourceFailure, OriginalHostSourceFailureCause,
    OriginalHostSourcePending, OriginalHostSourceReceipt, WorkingMemoryError,
};
use safemlx::{PreparedAllocationOwner, PreparedAllocationOwnerCause};
use std::{cell::Cell, sync::Mutex};

type Registration = NativeStorageRegistration<StorageIdentity>;
type Attachment = PreparedAllocationOwner<Registration>;
type Failure = OriginalHostSourceFailure<StorageIdentity, Completed, Attachment, SourceCause>;
type RetiredFailure = OriginalHostSourceFailure<StorageIdentity, (), Attachment, SourceCause>;
type Pending = OriginalHostSourcePending<StorageIdentity, Completed, Attachment, SourceCause>;

/// Kept in the actual scan row before submission; the pending source holds its
/// accepted receipt from before the first Host allocation through publication.
pub(super) struct Store {
    pending: Option<Pending>,
    completed: Option<Completed>,
}
struct Completed {
    destination: PreparedHostCopyDestination,
    ready: Option<ImmutableHostTransferBuffer>,
    allocation: Cell<Option<AllocationInfo>>,
}
struct Observation<'a> {
    buffer: &'a ImmutableHostTransferBuffer,
    allocation: AllocationInfo,
}
#[derive(thiserror::Error)]
pub(crate) enum SourceCause {
    #[error(transparent)]
    Input(safemlx::PreparedInputCause),
    #[error(transparent)]
    Arena(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)]
    Observation(safemlx::HostTransferMetadataError),
    #[error("Host source attachment: {cause}")]
    Attachment {
        #[source]
        cause: PreparedAllocationOwnerCause,
        registration: Option<Registration>,
    },
    #[error("Host store has not completed immutable publication")]
    State,
}
impl std::fmt::Debug for SourceCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct SourceError {
    #[source]
    cause: OriginalHostSourceFailureCause<SourceCause>,
    retained: Mutex<RetiredFailure>,
}
struct Copy<'a> {
    plan: PreparedHostTransferPlan<'a>,
    custody: &'a OriginalHostSourceCustody,
}
impl OriginalHostSourceConstruction for Copy<'_> {
    type Key = StorageIdentity;
    type Completed = Completed;
    type Output = Completed;
    type Attachment = Attachment;
    type Error = SourceCause;
    type Observation<'a> = Observation<'a>;
    fn source_custody(&self) -> OriginalHostSourceCustody {
        self.custody.clone()
    }
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError> {
        storage_bytes(&self.plan)
    }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Completed, SourceCause> {
        let quota = PreparedSubmissionGraphQuota::try_new(self.plan.metadata_bytes(), receipt)
            .map_err(|cause| SourceCause::Arena(cause.cause()))?;
        let arena = PreparedInputArena::try_allocate(quota)
            .map_err(|cause| SourceCause::Arena(cause.cause()))?;
        Ok(Completed {
            destination: self
                .plan
                .construct_copy_destination(&arena)
                .map_err(SourceCause::Input)?,
            ready: None,
            allocation: Cell::new(None),
        })
    }
    fn observe(completed: &Completed) -> Result<Observation<'_>, SourceCause> {
        let buffer = completed.ready.as_ref().ok_or(SourceCause::State)?;
        let allocation = buffer
            .try_allocation_info()
            .map_err(SourceCause::Observation)?;
        completed.allocation.set(Some(allocation));
        Ok(Observation { buffer, allocation })
    }
    fn describe(observation: &Observation<'_>) -> Option<(StorageIdentity, u64)> {
        (observation.allocation.bytes() != 0).then(|| {
            (
                StorageIdentity::Native(observation.allocation.identity()),
                observation.allocation.bytes() as u64,
            )
        })
    }
    fn prepare_attachment(registration: Registration) -> Result<Attachment, SourceCause> {
        Attachment::try_new(registration).map_err(|error| {
            let (cause, registration) = error.into_parts();
            SourceCause::Attachment {
                cause,
                registration: Some(registration),
            }
        })
    }
    fn registration(attachment: &Attachment) -> &Registration {
        attachment.owner()
    }
    fn attach(
        observation: Observation<'_>,
        attachment: Attachment,
    ) -> Result<(), (SourceCause, Attachment)> {
        observation
            .buffer
            .try_attach_prepared_allocation_owner(attachment)
            .map_err(|error| {
                let (cause, attachment) = error.into_parts();
                (
                    SourceCause::Attachment {
                        cause,
                        registration: None,
                    },
                    attachment,
                )
            })
    }
    fn into_output(completed: Completed) -> Completed {
        completed
    }
}
pub(super) fn storage_bytes(
    plan: &PreparedHostTransferPlan<'_>,
) -> Result<(u64, u64), WorkingMemoryError> {
    let overflow = || WorkingMemoryError::Overflow;
    let arena = PreparedInputArena::layout::<OriginalHostSourceReceipt>(plan.metadata_bytes())
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
    let attachment = Attachment::layout();
    let parts = [
        arena.total_bytes().ok_or_else(overflow)?,
        plan.control_bytes().ok_or_else(overflow)?,
        PreparedHostCopyDestination::control_bytes().ok_or_else(overflow)?,
        usize::try_from(OriginalHostSourceBank::publication_control_bytes::<Copy<'_>>(0)?)
            .map_err(|_| overflow())?,
        attachment.allocation_bytes().ok_or_else(overflow)?,
        attachment.preparation_control_bytes(),
        attachment.preparation_failure_bytes(),
        attachment.attachment_failure_bytes(),
        attachment.original_attachment_control_bytes(),
        size_of::<Copy<'_>>(),
        size_of::<Observation<'_>>(),
        size_of::<Completed>(),
        size_of::<Store>(),
        size_of::<SourceError>(),
        size_of::<SourceCause>(),
        size_of::<Result<Store, SourceError>>(),
        size_of::<Mutex<Failure>>(),
        size_of::<Mutex<RetiredFailure>>(),
        size_of::<Result<AllocationInfo, safemlx::HostTransferMetadataError>>(),
        size_of::<OriginalHostSourceReceipt>(),
    ];
    let controls = parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
        .ok_or_else(overflow)?;
    let total = controls
        .checked_add(plan.backing_bytes())
        .ok_or_else(overflow)?;
    Ok((
        u64::try_from(total).map_err(|_| overflow())?,
        u64::try_from(plan.backing_bytes()).map_err(|_| overflow())?,
    ))
}
fn failure(mut retained: Failure) -> SourceError {
    let cause = std::mem::replace(
        retained.cause_mut(),
        OriginalHostSourceFailureCause::Memory(WorkingMemoryError::AlreadyStarted),
    );
    // begin failures are unsubmitted; finish is called only after the exact
    // positive copy wait. The Model root union still retains submitted aliases
    // and the scan row retains their cache-pool reservation. The escaping error
    // keeps typed cause/receipt/registration, never an uncharged native prefix.
    let retained = retained.map_completed(drop);
    SourceError {
        cause,
        retained: Mutex::new(retained),
    }
}
impl Store {
    pub(super) fn begin(
        bank: &mut OriginalHostSourceBank,
        plan: PreparedHostTransferPlan<'_>,
        custody: &OriginalHostSourceCustody,
    ) -> Result<Self, SourceError> {
        let pending = bank.begin(Copy { plan, custody }).map_err(|error| {
            let (unstarted, retained) = error.into_parts();
            let error = failure(retained);
            drop(unstarted);
            error
        })?;
        Ok(Self {
            pending: Some(pending),
            completed: None,
        })
    }
    pub(super) fn destination(&mut self) -> &mut PreparedHostCopyDestination {
        &mut self
            .pending
            .as_mut()
            .expect("pending store")
            .completed_mut()
            .destination
    }
    pub(super) fn finish(&mut self) -> Result<(), SourceError> {
        let pending = self.pending.take().expect("one immutable publication");
        self.completed = Some(pending.finish::<Copy<'_>>().map_err(failure)?);
        Ok(())
    }
    pub(super) fn set_ready(&mut self, buffer: ImmutableHostTransferBuffer) {
        self.pending
            .as_mut()
            .expect("pending store")
            .completed_mut()
            .ready = Some(buffer);
    }
    pub(super) fn take_ready(&mut self) -> ImmutableHostTransferBuffer {
        self.completed
            .as_mut()
            .expect("published store")
            .ready
            .take()
            .expect("one canonical handoff")
    }
}
