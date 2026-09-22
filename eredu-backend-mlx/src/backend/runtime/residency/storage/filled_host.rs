//! The existing source transaction over one final, actually initialized host buffer.
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    HostSourcePeakSelection, NativeStorageRegistration, OriginalHostSourceConstruction,
    OriginalHostSourceFailure, OriginalHostSourceFailureCause, OriginalHostSourcePending,
    OriginalHostSourceReceipt, OriginalOperationMetadataCustody,
};
use eredu_runtime::working_memory::{
    OriginalHostSourceBank, OriginalHostSourceCustody, WorkingMemoryError,
};
use safemlx::{
    AllocationInfo, HostTransferMetadataError, PreparedAllocationOwner,
    PreparedAllocationOwnerCause,
};
use safemlx::{
    ImmutableHostTransferBuffer, PreparedHostTransferPlan, PreparedInputArena, PreparedInputCause,
    PreparedSubmissionGraphQuota,
};
use std::mem::{size_of, size_of_val};
use std::sync::Mutex;

type Registration = NativeStorageRegistration<StorageIdentity>;
type Attachment = PreparedAllocationOwner<Registration>;
type Failure = OriginalHostSourceFailure<StorageIdentity, Completed, Attachment, SourceCause>;
type RetiredFailure = OriginalHostSourceFailure<
    StorageIdentity,
    Option<ImmutableHostTransferBuffer>,
    Attachment,
    SourceCause,
>;
pub(crate) type Pending =
    OriginalHostSourcePending<StorageIdentity, Completed, Attachment, SourceCause>;
/// Writable only until the existing complete batch read succeeds. This type is
/// unforgeable outside this producer; observation refuses its writable state.
pub(crate) struct Completed {
    filling: Option<safemlx::PreparedHostTransferWriter>,
    ready: Option<ImmutableHostTransferBuffer>,
    allocation: std::cell::Cell<Option<AllocationInfo>>,
    custody: OriginalOperationMetadataCustody,
}
/// The existing source transaction creates this only after successful canonical
/// attachment. Its scalar facts grant no birth and retain no extra native owner.
pub(crate) struct PublishedHostSource {
    buffer: ImmutableHostTransferBuffer,
    allocation: AllocationInfo,
    proof: Option<super::PublishedAllocation>,
    custody: OriginalOperationMetadataCustody,
}
impl PublishedHostSource {
    pub(crate) fn buffer(&self) -> &ImmutableHostTransferBuffer {
        &self.buffer
    }
    pub(crate) fn allocation(&self) -> AllocationInfo {
        self.allocation
    }
    pub(crate) fn into_parts(
        self,
    ) -> (
        ImmutableHostTransferBuffer,
        Option<super::PublishedAllocation>,
        OriginalOperationMetadataCustody,
    ) {
        (self.buffer, self.proof, self.custody)
    }
}
impl Completed {
    fn into_failed_owner(self) -> Option<ImmutableHostTransferBuffer> {
        // The exclusive writer queues native ownership when a worker-side
        // failure drops it. No native destruction runs on the I/O thread;
        // already-frozen owners remain in the precise source failure.
        let Self { filling, ready, .. } = self;
        drop(filling);
        ready
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        self.filling
            .as_mut()
            .expect("uninitialized source")
            .as_bytes_mut()
    }
    pub(crate) fn freeze(&mut self) -> Result<(), SourceCause> {
        if self.ready.is_some() && self.filling.is_none() {
            return Ok(());
        }
        match self
            .filling
            .take()
            .ok_or(SourceCause::Identity)?
            .try_return()
        {
            Ok(buffer) => {
                self.ready = Some(buffer.freeze());
                Ok(())
            }
            Err(writer) => {
                self.filling = Some(writer);
                Err(SourceCause::Identity)
            }
        }
    }

    /// The caller's native recovery owns the destination before submission.
    /// Only this producer can convert its exact I/O writer into that native
    /// destination; no tensor or completion becomes part of the Send handoff.
    pub(crate) fn copy_from_array(
        &mut self,
        source: &safemlx::Array,
        stream: &safemlx::Stream,
        observer: &safemlx::OriginalScopeObserver,
        retained: &mut Option<safemlx::PreparedHostCopyDestination>,
    ) -> Result<(), SourceCause> {
        if retained.is_some() || self.ready.is_some() {
            return Err(SourceCause::Identity);
        }
        let writer = self.filling.take().ok_or(SourceCause::Identity)?;
        match safemlx::PreparedHostCopyDestination::from_writer(writer) {
            Ok(destination) => *retained = Some(destination),
            Err(writer) => {
                self.filling = Some(writer);
                return Err(SourceCause::Identity);
            }
        }
        let destination = retained
            .as_mut()
            .expect("retained before native submission");
        destination.submit(source, stream, observer)?;
        destination.synchronize()?;
        self.ready = Some(destination.take_completed()?);
        Ok(())
    }
}
pub(crate) fn native_copy_control_bytes() -> Option<usize> {
    safemlx::PreparedHostCopyDestination::control_bytes()?.checked_add(size_of::<(
        &mut Completed,
        &safemlx::Array,
        &safemlx::Stream,
        &safemlx::OriginalScopeObserver,
        &mut Option<safemlx::PreparedHostCopyDestination>,
        Result<(), SourceCause>,
    )>())
}
#[derive(thiserror::Error)]
pub(crate) enum SourceCause {
    #[error("source constructor: {0}")]
    Native(#[from] PreparedInputCause),
    #[error("source arena: {0}")]
    Arena(#[from] safemlx::SubmissionGraphQuotaCause),
    #[error("source destination: {0}")]
    Destination(#[from] safemlx::error::Exception),
    #[error("source native store: {0}")]
    PreparedCopy(#[from] safemlx::PreparedHostCopyError),
    #[error("source observation: {0}")]
    Observation(#[from] HostTransferMetadataError),
    #[error("source attachment: {cause}")]
    Attachment {
        #[source]
        cause: PreparedAllocationOwnerCause,
        registration: Option<Registration>,
    },
    #[error("source geometry mismatch")]
    Identity,
}
impl std::fmt::Debug for SourceCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
/// Move the precise cause out, while the same completed/attachment prefix stays
/// retained. Mutex makes the Send-only native preparation safely shareable as
/// an error source; its payload is never inspected while a native lock is held.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct SourceError {
    #[source]
    cause: OriginalHostSourceFailureCause<SourceCause>,
    retained: Mutex<RetiredFailure>,
}
/// The actual selected constructor's retirement permit. A source bank may
/// select a peak only when this permit authenticates its same live owner.
pub(crate) trait HostFillPermit: Sized {
    type Owner: Send + 'static;
    fn validate(
        &self,
        selection: HostSourcePeakSelection,
        owner: usize,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError>;
    fn create_quota(
        self,
        bytes: usize,
        receipt: OriginalHostSourceReceipt,
    ) -> Result<PreparedSubmissionGraphQuota<Self::Owner>, SourceCause>;
}
/// Cumulative constructor custody, used only by a bank that already admitted
/// this particular source. It cannot satisfy a peak-backed bank's permit.
pub(crate) struct Cumulative;
impl HostFillPermit for Cumulative {
    type Owner = OriginalHostSourceReceipt;
    fn validate(
        &self,
        _: HostSourcePeakSelection,
        _: usize,
        _: u64,
    ) -> Result<(), WorkingMemoryError> {
        Err(WorkingMemoryError::IdentityMismatch)
    }
    fn create_quota(
        self,
        bytes: usize,
        receipt: OriginalHostSourceReceipt,
    ) -> Result<PreparedSubmissionGraphQuota<Self::Owner>, SourceCause> {
        PreparedSubmissionGraphQuota::try_new(bytes, receipt)
            .map_err(|error| SourceCause::Arena(error.cause()))
    }
}
struct Copy<'a, P> {
    plan: PreparedHostTransferPlan<'a>,
    permit: P,
    controls: &'a OriginalHostSourceCustody,
}
struct Observation<'a> {
    buffer: &'a ImmutableHostTransferBuffer,
    info: AllocationInfo,
}
impl<P: HostFillPermit> OriginalHostSourceConstruction for Copy<'_, P> {
    type Key = StorageIdentity;
    type Completed = Completed;
    type Output = PublishedHostSource;
    type Attachment = Attachment;
    type Error = SourceCause;
    type Observation<'a> = Observation<'a>;
    fn source_custody(&self) -> eredu_runtime::working_memory::OriginalHostSourceCustody {
        self.controls.clone().into()
    }
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError> {
        let controls = control_bytes::<P>(&self.plan)?;
        let total = controls
            .checked_add(self.plan.backing_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok((total as u64, self.plan.backing_bytes() as u64))
    }
    fn validate_peak_permit(
        &self,
        selection: HostSourcePeakSelection,
        owner: usize,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        self.permit.validate(selection, owner, bytes)
    }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Self::Completed, Self::Error> {
        // One move consumes the exact per-buffer split permit and the bank's
        // accepted constructor receipt before the first native allocation.
        let quota = self
            .permit
            .create_quota(self.plan.metadata_bytes(), receipt)?;
        let arena = PreparedInputArena::try_allocate(quota).map_err(|error| {
            let cause = error.cause();
            drop(error);
            SourceCause::Arena(cause)
        })?;
        Ok(Completed {
            filling: Some(self.plan.construct_writer(&arena)?),
            ready: None,
            allocation: std::cell::Cell::new(None),
            custody: self.controls.metadata_custody(),
        })
    }
    fn observe(completed: &Self::Completed) -> Result<Self::Observation<'_>, Self::Error> {
        let buffer = completed.ready.as_ref().ok_or(SourceCause::Identity)?;
        let info = buffer.try_allocation_info()?;
        completed.allocation.set(Some(info));
        Ok(Observation { buffer, info })
    }
    fn describe(observation: &Self::Observation<'_>) -> Option<(StorageIdentity, u64)> {
        if observation.info.bytes() == 0 {
            None
        } else {
            Some((
                StorageIdentity::Native(observation.info.identity()),
                observation.info.bytes() as u64,
            ))
        }
    }
    fn prepare_attachment(registration: Registration) -> Result<Attachment, Self::Error> {
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
        observation: Self::Observation<'_>,
        attachment: Attachment,
    ) -> Result<(), (Self::Error, Attachment)> {
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
    fn into_output(completed: Self::Completed) -> Self::Output {
        let allocation = completed
            .allocation
            .get()
            .expect("observed completed allocation");
        PublishedHostSource {
            buffer: completed.ready.expect("observed completed source"),
            allocation,
            // Zero-capacity sources have no canonical row or attachment.
            proof: (allocation.bytes() != 0)
                .then(|| super::PublishedAllocation::attached(allocation)),
            custody: completed.custody,
        }
    }
}
pub(crate) fn control_bytes<P: HostFillPermit>(
    plan: &PreparedHostTransferPlan<'_>,
) -> Result<usize, WorkingMemoryError> {
    let overflow = || WorkingMemoryError::Overflow;
    let arena = PreparedInputArena::layout::<P::Owner>(plan.metadata_bytes())
        .map_err(|_| WorkingMemoryError::UnknownBound)?;
    let attachment = Attachment::layout();
    let fixed = [
        arena.total_bytes().ok_or_else(overflow)?,
        plan.control_bytes().ok_or_else(overflow)?,
        plan.writer_control_bytes().ok_or_else(overflow)?,
        usize::try_from(OriginalHostSourceBank::publication_control_bytes::<
            Copy<'_, P>,
        >(0)?)
        .map_err(|_| overflow())?,
        attachment.allocation_bytes().ok_or_else(overflow)?,
        attachment.preparation_control_bytes(),
        attachment.preparation_failure_bytes(),
        attachment.attachment_failure_bytes(),
        attachment.original_attachment_control_bytes(),
        size_of::<Copy<'_, P>>(),
        size_of::<Observation<'_>>(),
        size_of::<SourceError>(),
        size_of::<SourceCause>(),
        size_of::<Result<PublishedHostSource, SourceError>>(),
        size_of::<Result<Pending, SourceError>>(),
        size_of::<Mutex<Failure>>(),
        size_of::<RetiredFailure>(),
        size_of::<Mutex<RetiredFailure>>(),
        size_of::<Result<AllocationInfo, HostTransferMetadataError>>(),
        size_of::<Result<AllocationInfo, safemlx::error::Exception>>(),
        size_of::<Result<u64, safemlx::error::Exception>>(),
        size_of::<Result<usize, safemlx::error::Exception>>(),
        size_of::<(&ImmutableHostTransferBuffer, u64, usize)>(),
        size_of::<Completed>(),
        size_of::<PublishedHostSource>(),
        size_of::<OriginalOperationMetadataCustody>(),
        size_of::<(
            ImmutableHostTransferBuffer,
            Option<super::PublishedAllocation>,
            OriginalOperationMetadataCustody,
        )>(),
        size_of::<AllocationInfo>(),
        size_of::<Option<super::PublishedAllocation>>(),
        size_of::<&AllocationInfo>(),
        size_of::<bool>(),
        size_of::<&PublishedHostSource>(),
        size_of::<Option<safemlx::PreparedHostTransferWriter>>(),
        size_of::<Option<ImmutableHostTransferBuffer>>(),
        size_of::<OriginalHostSourceReceipt>(),
        size_of::<P::Owner>(),
    ];
    fixed
        .into_iter()
        .try_fold(size_of_val(&fixed), usize::checked_add)
        .ok_or_else(overflow)
}
pub(crate) fn begin<'a, P: HostFillPermit>(
    bank: &mut OriginalHostSourceBank,
    plan: PreparedHostTransferPlan<'a>,
    permit: P,
    controls: &'a OriginalHostSourceCustody,
) -> Result<Pending, SourceError> {
    bank.begin(Copy {
        plan,
        permit,
        controls,
    })
    .map_err(|error| {
        let (unstarted, retained) = error.into_parts();
        let error = failure(retained);
        drop(unstarted);
        error
    })
}
pub(crate) fn finish(pending: Pending) -> Result<PublishedHostSource, SourceError> {
    // Permit identity was consumed and validated before allocation. Every
    // producer uses this exact completed/observation/attachment worker; finish
    // cannot reopen construction or acquire a fresh source grant.
    pending.finish::<Copy<'_, Cumulative>>().map_err(failure)
}
fn failure(mut retained: Failure) -> SourceError {
    let cause = std::mem::replace(
        retained.cause_mut(),
        OriginalHostSourceFailureCause::Memory(WorkingMemoryError::AlreadyStarted),
    );
    let retained = retained.map_completed(Completed::into_failed_owner);
    SourceError {
        cause,
        retained: Mutex::new(retained),
    }
}
