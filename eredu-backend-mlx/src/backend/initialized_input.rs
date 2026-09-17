//! Canonical registered publication for the actual prepared native input producer.
use crate::backend::error::Error;
use safemlx::{Array,PreparedInputPlan};
use std::mem::{size_of,size_of_val};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{NativeStorageRegistration,
    OriginalHostSourceBank, OriginalHostSourceConstruction, OriginalHostSourceCustody,
    OriginalHostSourceFailure, OriginalHostSourceFailureCause, OriginalHostSourceReceipt,
    WorkingMemoryError};
use safemlx::{PreparedInputArena, PreparedInputLeaf, PreparedSubmissionGraphQuota,
    PreparedAllocationOwner, PreparedAllocationOwnerCause, ImmutableSourceWitness};
use std::sync::Mutex;

type Registration = NativeStorageRegistration<StorageIdentity>;
type Attachment = PreparedAllocationOwner<Registration>;
type Failure = OriginalHostSourceFailure<StorageIdentity, Array, Attachment, InputCause>;
type RetiredFailure = OriginalHostSourceFailure<StorageIdentity, (), Attachment, InputCause>;
#[derive(thiserror::Error)]
enum InputCause {
    #[error(transparent)] Input(safemlx::PreparedInputCause),
    #[error(transparent)] Arena(safemlx::SubmissionGraphQuotaCause),
    #[error(transparent)] Observation(safemlx::OriginalBufferCause),
    #[error("initialized input source attachment: {cause}")]
    Attachment { #[source] cause: PreparedAllocationOwnerCause, registration: Option<Registration> },
    #[error("initialized input source has no completed immutable allocation")] State,
}
impl std::fmt::Debug for InputCause { fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { std::fmt::Display::fmt(self,f) } }
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct InputFailure {
    #[source] cause: OriginalHostSourceFailureCause<InputCause>,
    retained: Mutex<RetiredFailure>,
}
struct Input<'a> { plan: PreparedInputPlan<'a>, custody: &'a OriginalHostSourceCustody, caller_controls:usize }
impl OriginalHostSourceConstruction for Input<'_> {
    type Key = StorageIdentity;
    type Completed = Array;
    type Output = Array;
    type Attachment = Attachment;
    type Error = InputCause;
    type Observation<'a> = ImmutableSourceWitness<'a>;
    fn source_custody(&self) -> OriginalHostSourceCustody { self.custody.clone() }
    fn storage_bytes(&self) -> Result<(u64,u64),WorkingMemoryError> { storage_bytes(&self.plan,self.caller_controls) }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Array,InputCause> {
        let wrapper = PreparedInputLeaf::array_layout().map_err(InputCause::Input)?;
        let metadata = self.plan.metadata_bytes().checked_add(wrapper.metadata_bytes()).ok_or(InputCause::State)?;
        let quota = PreparedSubmissionGraphQuota::try_new(metadata, receipt)
            .map_err(|failure| InputCause::Arena(failure.cause()))?;
        let arena = PreparedInputArena::try_allocate(quota)
            .map_err(|failure| InputCause::Arena(failure.cause()))?;
        let leaf = self.plan.construct(&arena).map_err(InputCause::Input)?;
        leaf.try_source_array().map_err(InputCause::Input)
    }
    fn observe(completed: &Array) -> Result<ImmutableSourceWitness<'_>,InputCause> {
        match completed.inspect_immutable_source().map_err(InputCause::Observation)? {
            safemlx::ImmutableSourceInspection::Allocation(value) => Ok(value),
            _ => Err(InputCause::State),
        }
    }
    fn describe(observation: &ImmutableSourceWitness<'_>) -> Option<(StorageIdentity,u64)> {
        let allocation=observation.allocation();
        Some((StorageIdentity::Native(allocation.identity()),allocation.bytes() as u64))
    }
    fn prepare_attachment(registration: Registration) -> Result<Attachment,InputCause> {
        Attachment::try_new(registration).map_err(|failure| {
            let (cause,registration)=failure.into_parts();
            InputCause::Attachment {cause,registration:Some(registration)}
        })
    }
    fn registration(attachment: &Attachment) -> &Registration { attachment.owner() }
    fn attach(observation: ImmutableSourceWitness<'_>, attachment: Attachment)
        -> Result<(),(InputCause,Attachment)> {
        observation.try_attach(attachment).map_err(|failure| {
            let (cause,attachment)=failure.into_parts();
            (InputCause::Observation(cause),attachment)
        })
    }
    fn into_output(completed: Array) -> Array { completed }
}
pub(crate) fn storage_bytes(plan: &PreparedInputPlan<'_>,caller_controls:usize) -> Result<(u64,u64),WorkingMemoryError> {
    let overflow=||WorkingMemoryError::Overflow;
    let wrapper=PreparedInputLeaf::array_layout().map_err(|_|WorkingMemoryError::UnknownBound)?;
    let metadata=plan.metadata_bytes().checked_add(wrapper.metadata_bytes()).ok_or_else(overflow)?;
    let arena=PreparedInputArena::layout::<OriginalHostSourceReceipt>(metadata)
        .map_err(|_|WorkingMemoryError::UnknownBound)?;
    let attachment=Attachment::layout();
    let parts=[arena.total_bytes().ok_or_else(overflow)?, plan.control_bytes(), wrapper.control_bytes(),
        usize::try_from(OriginalHostSourceBank::publication_control_bytes::<Input<'_>>(0)?).map_err(|_|overflow())?,
        attachment.allocation_bytes().ok_or_else(overflow)?, attachment.preparation_control_bytes(),
        attachment.preparation_failure_bytes(), attachment.attachment_failure_bytes(),
        attachment.original_attachment_control_bytes(),
        ImmutableSourceWitness::inspection_control_bytes().ok_or_else(overflow)?,
        eredu_core::SharedBackendFailure::control_bytes::<InputFailure>().ok_or_else(overflow)?,
        size_of::<Input<'_>>(),size_of::<Array>(),size_of::<PreparedInputLeaf>(),
        size_of::<InputCause>(),size_of::<InputFailure>(),size_of::<Failure>(),size_of::<RetiredFailure>(),
        size_of::<OriginalHostSourceCustody>(),size_of::<OriginalHostSourceReceipt>(),
        size_of::<Result<Array,Error>>(),size_of::<Result<Array,InputCause>>(),size_of::<Result<Attachment,InputCause>>(),
        size_of::<Result<PreparedInputArena,safemlx::SubmissionGraphQuotaError<PreparedSubmissionGraphQuota<OriginalHostSourceReceipt>>>>(),
        size_of::<Result<PreparedSubmissionGraphQuota<OriginalHostSourceReceipt>,safemlx::SubmissionGraphQuotaError<OriginalHostSourceReceipt>>>(),
        size_of::<Result<PreparedInputLeaf,safemlx::PreparedInputCause>>(),
        size_of::<(&mut OriginalHostSourceBank,&OriginalHostSourceCustody,PreparedInputPlan<'_>,usize)>(),
        size_of::<usize>(), caller_controls,
    ];
    let controls=parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add).ok_or_else(overflow)?;
    let total=controls.checked_add(plan.backing_bytes()).ok_or_else(overflow)?;
    Ok((u64::try_from(total).map_err(|_|overflow())?,u64::try_from(plan.backing_bytes()).map_err(|_|overflow())?))
}
/// Publishes one initialized native allocation through the original accepted
/// bank. Failure retains its spent source/registration custody; no new allowance
/// or current-scope inference is used by this worker.
pub(crate) fn construct(bank:&mut OriginalHostSourceBank,custody:&OriginalHostSourceCustody,
    plan:PreparedInputPlan<'_>,caller_controls:usize)->Result<Array,Error> {
    bank.construct(Input{plan,custody,caller_controls})
        .map_err(|failure| {
            let (unstarted,mut retained)=failure.into_parts();
            drop(unstarted);
            let cause=std::mem::replace(retained.cause_mut(),OriginalHostSourceFailureCause::Memory(WorkingMemoryError::AlreadyStarted));
            // This eager leaf is either unsubmitted or already initialized; its
            // native storage retires before escaped source/registration custody.
            let retained=retained.map_completed(drop);
            Error::retained_original(eredu_core::SharedBackendFailure::new(eredu_core::BackendFailureKind::Other,
                InputFailure{cause,retained:Mutex::new(retained)}),false)
        })
}
