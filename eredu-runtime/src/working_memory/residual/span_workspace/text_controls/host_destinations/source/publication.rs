//! One selected actual immutable construction and its prepaid-host publication.
use super::*;
use crate::working_memory::{
    NativeStorageRegistration,
    storage::{native_publication::PreparedNativePublication, prepaid::PrepaidHostOrigin},
};
use std::sync::OnceLock;

/// Actual backend source producer consumed by the same accepted host bank.
/// Implementations bind the retained selected source/recipe and actual copy plan;
/// constructors belong to the backend's selected mechanism, not source callers.
/// No method may substitute observed/caller-supplied capacity for the prepared
/// allocation strategy. The completed owner must be unforgeable from raw keys or
/// arbitrary arrays. All provider calls occur outside Usage; the producer keeps
/// native/source failure owners through their real completion/retirement boundary.
pub trait OriginalHostSourceConstruction: Sized {
    type Key: Clone + Ord + Send + Sync + 'static;
    type Completed: 'static;
    type Output;
    type Attachment: 'static;
    type Error: 'static;
    /// Borrows the completed owner independently of the consumed constructor.
    type Observation<'a>;

    fn source_custody(&self) -> OriginalHostSourceCustody;
    /// Complete C_new+B from this actual selected constructor, and its separate
    /// B. This is a requirement consumed below, never a grant or origin witness.
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError>;
    /// For a peak-backed bank, authenticate the actual move-only permit against
    /// its selected live owner before the constructor consumes that permit.
    /// Ordinary cumulative producers cannot enter this path by default.
    fn validate_peak_permit(
        &self,
        _selection: HostSourcePeakSelection,
        _owner: usize,
        _bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        Err(WorkingMemoryError::IdentityMismatch)
    }
    /// Returns only this invocation's unforgeable actual constructed owner.
    /// observe must refuse until its initialization and immutable handoff finish.
    /// Receipt must enter its retirement owner before the first allocation.
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Self::Completed, Self::Error>;
    fn observe(completed: &Self::Completed) -> Result<Self::Observation<'_>, Self::Error>;
    /// Only a positively allocation-free copy returns None.
    fn describe(observation: &Self::Observation<'_>) -> Option<(Self::Key, u64)>;
    fn prepare_attachment(
        registration: NativeStorageRegistration<Self::Key>,
    ) -> Result<Self::Attachment, Self::Error>;
    fn registration(attachment: &Self::Attachment) -> &NativeStorageRegistration<Self::Key>;
    fn attach(
        observation: Self::Observation<'_>,
        attachment: Self::Attachment,
    ) -> Result<(), (Self::Error, Self::Attachment)>;
    fn into_output(completed: Self::Completed) -> Self::Output;
}

#[derive(Debug)]
pub enum OriginalHostSourceFailureCause<E> {
    Funding(OriginalHostSourceError),
    Memory(WorkingMemoryError),
    Native(E),
}
impl<E: std::fmt::Display> std::fmt::Display for OriginalHostSourceFailureCause<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Funding(c) => c.fmt(f),
            Self::Memory(c) => c.fmt(f),
            Self::Native(c) => c.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for OriginalHostSourceFailureCause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Funding(c) => c,
            Self::Memory(c) => c,
            Self::Native(c) => c,
        })
    }
}

/// Completed-copy/attachment/registration prefix, followed by account custody.
/// No failure rolls back visible native attachments or canonical rows.
pub struct OriginalHostSourceFailure<K: Clone + Ord + Send + Sync + 'static, C, A, E> {
    cause: OriginalHostSourceFailureCause<E>,
    completed: Option<C>,
    attachment: Option<A>,
    registry: Option<PreparedNativePublication<K>>,
    controls: Option<OriginalHostSourceCustody>,
}
impl<K: Clone + Ord + Send + Sync + 'static, C, A, E> OriginalHostSourceFailure<K, C, A, E> {
    pub fn cause(&self) -> &OriginalHostSourceFailureCause<E> {
        &self.cause
    }
    pub fn cause_mut(&mut self) -> &mut OriginalHostSourceFailureCause<E> {
        &mut self.cause
    }
    pub fn completed(&self) -> Option<&C> {
        self.completed.as_ref()
    }
    /// Converts or retires the failed producer's completed prefix while the
    /// same attachment, registry and account custody remain retained. This
    /// grants no successful publication or renewed construction authority.
    pub fn map_completed<D>(
        mut self,
        convert: impl FnOnce(C) -> D,
    ) -> OriginalHostSourceFailure<K, D, A, E> {
        let completed = self.completed.take().map(convert);
        OriginalHostSourceFailure {
            cause: self.cause,
            completed,
            attachment: self.attachment,
            registry: self.registry,
            controls: self.controls,
        }
    }
}
impl<K: Clone + Ord + Send + Sync + 'static, C, A, E> std::fmt::Debug
    for OriginalHostSourceFailure<K, C, A, E>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalHostSourceFailure")
            .field("completed", &self.completed.is_some())
            .field("attachment", &self.attachment.is_some())
            .finish_non_exhaustive()
    }
}

/// Before native entry the exact producer/input is returned alongside custody.
/// After entry only the producer's own failure/completed prefix is retained.
pub struct OriginalHostSourceRefusal<M: OriginalHostSourceConstruction> {
    unstarted: Option<M>,
    retained: OriginalHostSourceFailure<M::Key, M::Completed, M::Attachment, M::Error>,
}
impl<M: OriginalHostSourceConstruction> std::fmt::Debug for OriginalHostSourceRefusal<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalHostSourceRefusal")
            .field("unstarted", &self.unstarted.is_some())
            .field("retained", &self.retained)
            .finish()
    }
}
impl<M: OriginalHostSourceConstruction> OriginalHostSourceRefusal<M> {
    pub fn into_parts(
        self,
    ) -> (
        Option<M>,
        OriginalHostSourceFailure<M::Key, M::Completed, M::Attachment, M::Error>,
    ) {
        (self.unstarted, self.retained)
    }
}

impl OriginalHostSourceBank {
    /// Existing one-step producers use the same split transaction internally.
    pub fn construct<M: OriginalHostSourceConstruction>(
        &mut self,
        producer: M,
    ) -> Result<M::Output, OriginalHostSourceRefusal<M>> {
        self.begin(producer)?
            .finish::<M>()
            .map_err(|retained| OriginalHostSourceRefusal {
                unstarted: None,
                retained,
            })
    }
    /// Debit and construct the actual final owner, retaining its receipt before
    /// allocation. This does not publish uninitialized storage. A producer may
    /// initialize multiple pending owners with its existing grouped read worker,
    /// then finish each only after its positive immutable observation is ready.
    pub fn begin<M: OriginalHostSourceConstruction>(
        &mut self,
        producer: M,
    ) -> Result<
        OriginalHostSourcePending<M::Key, M::Completed, M::Attachment, M::Error>,
        OriginalHostSourceRefusal<M>,
    > {
        let mut failure = OriginalHostSourceRefusal {
            unstarted: Some(producer),
            retained: OriginalHostSourceFailure {
                cause: OriginalHostSourceFailureCause::Memory(WorkingMemoryError::AlreadyStarted),
                completed: None,
                attachment: None,
                registry: None,
                controls: None,
            },
        };
        let result = (|| {
            let producer = failure.unstarted.as_ref().expect("unstarted source");
            if !self.belongs_to_source(&producer.source_custody()) {
                return Err(OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            let (total, backing) = producer
                .storage_bytes()
                .map_err(OriginalHostSourceFailureCause::Memory)?;
            if backing > total {
                return Err(OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            // This result is the sole success authority. Refused debit receipts
            // stay in Funding errors and cannot create PrepaidHostOrigin.
            let debit = match self.peak {
                Some(selection) => {
                    let owner = self
                        .peak_owner
                        .ok_or(OriginalHostSourceFailureCause::Memory(
                            WorkingMemoryError::IdentityMismatch,
                        ))?;
                    producer
                        .validate_peak_permit(selection, owner, backing)
                        .map_err(OriginalHostSourceFailureCause::Memory)?;
                    total - backing
                }
                None => total,
            };
            let receipt = self
                .try_debit(debit)
                .map_err(OriginalHostSourceFailureCause::Funding)?;
            failure.retained.controls = Some(self.controls.clone());
            let completed = failure
                .unstarted
                .take()
                .expect("one source call")
                .create(receipt)
                .map_err(OriginalHostSourceFailureCause::Native)?;
            failure.retained.completed = Some(completed);
            Ok(backing)
        })();
        match result {
            Ok(backing) => Ok(OriginalHostSourcePending {
                retained: failure.retained,
                backing,
                reservation: self.reservation.clone(),
            }),
            Err(cause) => {
                failure.retained.cause = cause;
                Err(failure)
            }
        }
    }

    /// Qualified finite controls of the shared source-publication worker.
    /// The actual backend additionally includes producer, native attachment and
    /// native/safe completion layouts in C_new before accepting its host bank.
    pub fn publication_control_bytes<M: OriginalHostSourceConstruction>(
        nested_key_bytes: u64,
    ) -> Result<u64, WorkingMemoryError> {
        let parts = [
            PreparedNativePublication::<M::Key>::qualified_control_bytes(1, nested_key_bytes)?,
            size_of::<OriginalHostSourceRefusal<M>>() as u64,
            size_of::<OriginalHostSourceFailure<M::Key, M::Completed, M::Attachment, M::Error>>()
                as u64,
            size_of::<Result<M::Output, OriginalHostSourceRefusal<M>>>() as u64,
            size_of::<Result<M::Completed, M::Error>>() as u64,
            size_of::<Option<(M::Key, u64)>>() as u64,
            size_of::<NativeStorageRegistration<M::Key>>() as u64,
            size_of::<PrepaidHostOrigin>() as u64,
            size_of::<(u64, u64)>() as u64,
            size_of::<OriginalHostSourcePending<M::Key, M::Completed, M::Attachment, M::Error>>()
                as u64,
            size_of::<
                Result<
                    OriginalHostSourcePending<M::Key, M::Completed, M::Attachment, M::Error>,
                    OriginalHostSourceRefusal<M>,
                >,
            >() as u64,
            size_of::<
                Result<
                    M::Output,
                    OriginalHostSourceFailure<M::Key, M::Completed, M::Attachment, M::Error>,
                >,
            >() as u64,
            size_of::<Option<WorkingMemoryReservation>>() as u64,
            size_of::<Option<&WorkingMemoryReservation>>() as u64,
            size_of::<OriginalHostSourceCustody>() as u64,
            size_of::<&OriginalHostSourceCustody>() as u64,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts) as u64, |a, b| {
                a.checked_add(b).ok_or(WorkingMemoryError::Overflow)
            })
    }
}

/// A move-only, already charged constructor result. No keys, capacities or
/// replacement origins can be supplied at finish. Native backing/receipt stays
/// inside the actual constructed owner, including on cancellation before finish.
pub struct OriginalHostSourcePending<K: Clone + Ord + Send + Sync + 'static, C, A, E> {
    retained: OriginalHostSourceFailure<K, C, A, E>,
    backing: u64,
    reservation: Option<WorkingMemoryReservation>,
}
impl<K: Clone + Ord + Send + Sync + 'static, C, A, E> OriginalHostSourcePending<K, C, A, E> {
    /// Only the selected producer's unforgeable owner is exposed for its actual
    /// grouped initialization. Its positive observation still gates publication.
    pub fn completed_mut(&mut self) -> &mut C {
        self.retained
            .completed
            .as_mut()
            .expect("constructed source")
    }
    /// Observe the initialized immutable owner and commit its one canonical row.
    pub fn finish<M>(mut self) -> Result<M::Output, OriginalHostSourceFailure<K, C, A, E>>
    where
        M: OriginalHostSourceConstruction<Key = K, Completed = C, Attachment = A, Error = E>,
    {
        let backing = self.backing;
        let controls = self
            .retained
            .controls
            .as_ref()
            .expect("accepted constructor custody");
        let result = (|| {
            let observed = M::observe(self.retained.completed.as_ref().expect("actual copy"))
                .map_err(OriginalHostSourceFailureCause::Native)?;
            let Some((key, bytes)) = M::describe(&observed) else {
                if backing != 0 {
                    return Err(OriginalHostSourceFailureCause::Memory(
                        WorkingMemoryError::IdentityMismatch,
                    ));
                }
                // No row is needed, but zero-copy success still revalidates
                // the exact accepted source/Work after native construction.
                controls
                    .validate_account(self.reservation.as_ref())
                    .map_err(OriginalHostSourceFailureCause::Memory)?;
                return Ok(());
            };
            if bytes != backing {
                return Err(OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            let origin =
                PrepaidHostOrigin::from_source_attempt(backing, controls.accounting());
            self.retained.registry = Some(
                PreparedNativePublication::prepare_source_exact(origin)
                    .map_err(OriginalHostSourceFailureCause::Memory)?,
            );
            self.retained.attachment = Some(
                M::prepare_attachment(NativeStorageRegistration {
                    registration: OnceLock::new(),
                    _raw: controls.accounting(),
                })
                .map_err(OriginalHostSourceFailureCause::Native)?,
            );
            // A provider may return only a fresh sidecar. Refuse a populated
            // owner before committing any canonical row; keep it intact.
            if M::registration(
                self.retained
                    .attachment
                    .as_ref()
                    .expect("prepared attachment"),
            )
            .registration
            .get()
            .is_some()
            {
                return Err(OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            let registry = self.retained.registry.as_mut().expect("prepared row");
            registry
                .push_source_birth(key, bytes)
                .map_err(OriginalHostSourceFailureCause::Memory)?;
            registry
                .publish_source(controls, self.reservation.as_ref())
                .map_err(OriginalHostSourceFailureCause::Memory)?;
            let registration = registry.take_input(0).ok_or_else(|| {
                OriginalHostSourceFailureCause::Memory(WorkingMemoryError::IdentityMismatch)
            })?;
            let attachment = self
                .retained
                .attachment
                .as_ref()
                .expect("prepared attachment");
            // No externally filled payload can replace this private pending row.
            if M::registration(attachment)
                .registration
                .set(registration)
                .is_err()
            {
                return Err(OriginalHostSourceFailureCause::Memory(
                    WorkingMemoryError::IdentityMismatch,
                ));
            }
            let attachment = self.retained.attachment.take().expect("one attach");
            if let Err((cause, attachment)) = M::attach(observed, attachment) {
                self.retained.attachment = Some(attachment);
                return Err(OriginalHostSourceFailureCause::Native(cause));
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(M::into_output(
                self.retained.completed.take().expect("successful source"),
            )),
            Err(cause) => {
                self.retained.cause = cause;
                Err(self.retained)
            }
        }
    }
}
