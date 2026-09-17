//! Actual GGUF copy producer; neutral runtime owns prepaid-H publication.
use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::{
    NativeStorageRegistration, OriginalHostSourceConstruction, OriginalHostSourceFailure,
    OriginalHostSourceFailureCause, WorkingMemoryError,
};
use safemlx::{CompletedOwnedHostCopy, ImmutableSourceInspection, PreparedAllocationOwner};

type Registration = NativeStorageRegistration<StorageIdentity>;
type Attachment = PreparedAllocationOwner<Registration>;
pub(super) type Failure<T> =
    OriginalHostSourceFailure<StorageIdentity, CompletedOwnedHostCopy, Attachment, CopyFailure<T>>;

enum InputFailure<T: safemlx::ArrayElement + Send + 'static> {
    Input {
        values: TypedValues<T>,
        preparation: Option<OwnedHostCopyPreparationError<SourceArenaCustody>>,
    },
    Copy(OwnedHostBufferCopyError<T, SourceArenaCustody, TypedValues<T>>),
}
pub(super) struct CopyFailure<T: safemlx::ArrayElement + Send + 'static> {
    cause: Option<GgufHostCopyCause>,
    input: Option<InputFailure<T>>,
    registration: Option<Registration>,
}
impl<T: safemlx::ArrayElement + Send + 'static> CopyFailure<T> {
    #[cfg(test)]
    pub(super) fn input(&self) -> &TypedValues<T> {
        match self.input.as_ref().expect("retained failed copy input") {
            InputFailure::Input { values, .. } => values,
            InputFailure::Copy(error) => error.buffer(),
        }
    }
    fn simple(cause: GgufHostCopyCause) -> Self {
        Self {
            cause: Some(cause),
            input: None,
            registration: None,
        }
    }
}
impl<T: safemlx::ArrayElement + Send + 'static> fmt::Debug for CopyFailure<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GgufSourceCopyFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl<T: safemlx::ArrayElement + Send + 'static> fmt::Display for CopyFailure<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Some(cause) => cause.fmt(f),
            None => f.write_str("source failure owner retained"),
        }
    }
}
impl<T: safemlx::ArrayElement + Send + 'static> std::error::Error for CopyFailure<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause
            .as_ref()
            .map(|c| c as &(dyn std::error::Error + 'static))
    }
}
struct Copy<'a, T: safemlx::ArrayElement + Send + 'static> {
    values: TypedValues<T>,
    plan: OwnedHostCopyPlan<'a, T>,
    storage: SourceCopyStorage,
    observer: &'a OriginalScopeObserver,
    custody: SourceCustody,
}
impl<T: safemlx::ArrayElement + Send + 'static> OriginalHostSourceConstruction for Copy<'_, T> {
    type Key = StorageIdentity;
    type Completed = CompletedOwnedHostCopy;
    type Output = Array;
    type Attachment = Attachment;
    type Error = CopyFailure<T>;
    type Observation<'a> = ImmutableSourceInspection<'a>;
    fn source_custody(&self) -> eredu_runtime::working_memory::OriginalHostSourceCustody {
        self.custody.controls.clone().into()
    }
    fn storage_bytes(&self) -> Result<(u64, u64), WorkingMemoryError> {
        Ok((
            self.storage
                .total_bytes()
                .map_err(|_| WorkingMemoryError::Overflow)?,
            u64::try_from(self.storage.backing).map_err(|_| WorkingMemoryError::Overflow)?,
        ))
    }
    fn create(self, receipt: OriginalHostSourceReceipt) -> Result<Self::Completed, Self::Error> {
        let Self {
            values,
            plan,
            observer,
            custody,
            ..
        } = self;
        let custody = SourceArenaCustody {
            source: custody,
            receipt: Some(receipt),
        };
        let arena =
            match PreparedSubmissionGraphQuota::try_new(plan.facts().metadata_bytes(), custody) {
                Ok(arena) => arena,
                Err(error) => {
                    let (cause, custody) = error.into_parts();
                    let failure = CopyFailure {
                        cause: Some(GgufHostCopyCause::Arena(cause)),
                        input: Some(InputFailure::Input {
                            values,
                            preparation: None,
                        }),
                        registration: None,
                    };
                    drop(custody);
                    return Err(failure);
                }
            };
        let ready = match plan.prepare(arena) {
            Ok(ready) => ready,
            Err(error) => {
                return Err(CopyFailure {
                    cause: Some(GgufHostCopyCause::Copy {
                        cause: error.cause(),
                        native: None,
                    }),
                    input: Some(InputFailure::Input {
                        values,
                        preparation: Some(error),
                    }),
                    registration: None,
                })
            }
        };
        ready
            .try_fill_owned_completed(values, observer)
            .map_err(|mut error| CopyFailure {
                cause: Some(GgufHostCopyCause::Copy {
                    cause: error.cause(),
                    native: error.take_native_source(),
                }),
                input: Some(InputFailure::Copy(error)),
                registration: None,
            })
    }
    fn observe(completed: &Self::Completed) -> Result<Self::Observation<'_>, Self::Error> {
        completed
            .observe()
            .map_err(|cause| CopyFailure::simple(GgufHostCopyCause::SourceNative(cause)))
    }
    fn describe(observation: &Self::Observation<'_>) -> Option<(StorageIdentity, u64)> {
        match observation {
            ImmutableSourceInspection::Empty => None,
            ImmutableSourceInspection::Allocation(witness) => {
                let info = witness.allocation();
                Some((
                    StorageIdentity::Native(info.identity()),
                    info.bytes() as u64,
                ))
            }
            ImmutableSourceInspection::Unknown => {
                unreachable!("completed-copy observation rejects unknown")
            }
        }
    }
    fn prepare_attachment(registration: Registration) -> Result<Attachment, Self::Error> {
        PreparedAllocationOwner::try_new(registration).map_err(|error| {
            let (cause, registration) = error.into_parts();
            CopyFailure {
                cause: Some(GgufHostCopyCause::SourceOwner(cause)),
                input: None,
                registration: Some(registration),
            }
        })
    }
    fn registration(attachment: &Attachment) -> &Registration {
        attachment.owner()
    }
    fn attach(
        observed: Self::Observation<'_>,
        attachment: Attachment,
    ) -> Result<(), (Self::Error, Attachment)> {
        let ImmutableSourceInspection::Allocation(witness) = observed else {
            return Err((
                CopyFailure::simple(GgufHostCopyCause::SourceNative(
                    safemlx::OriginalBufferCause::UncertifiedBacking,
                )),
                attachment,
            ));
        };
        witness.try_attach(attachment).map_err(|error| {
            let (cause, attachment) = error.into_parts();
            (
                CopyFailure::simple(GgufHostCopyCause::SourceNative(cause)),
                attachment,
            )
        })
    }
    fn into_output(completed: Self::Completed) -> Array {
        completed.into_array()
    }
}
pub(super) fn create<'a, T: safemlx::ArrayElement + Send + 'static>(
    bank: &mut OriginalHostSourceBank,
    plan: OwnedHostCopyPlan<'a, T>,
    values: TypedValues<T>,
    custody: SourceCustody,
    storage: SourceCopyStorage,
    observer: &'a OriginalScopeObserver,
) -> Result<Array, (GgufHostCopyCause, TypedFailure<T>)> {
    match bank.construct(Copy {
        plan,
        values,
        custody,
        storage,
        observer,
    }) {
        Ok(array) => Ok(array),
        Err(error) => {
            let (unstarted, mut retained) = error.into_parts();
            let original = std::mem::replace(
                retained.cause_mut(),
                OriginalHostSourceFailureCause::Memory(WorkingMemoryError::AlreadyStarted),
            );
            let cause = match original {
                OriginalHostSourceFailureCause::Funding(cause) => {
                    GgufHostCopyCause::SourceFunding(cause)
                }
                OriginalHostSourceFailureCause::Memory(cause) => {
                    GgufHostCopyCause::SourcePublication(cause)
                }
                OriginalHostSourceFailureCause::Native(mut owner) => {
                    let cause = owner.cause.take().expect("one diagnostic extraction");
                    *retained.cause_mut() = OriginalHostSourceFailureCause::Native(owner);
                    cause
                }
            };
            let owner = if let Some(Copy { values, .. }) = unstarted {
                drop(retained);
                TypedFailure::Input {
                    values,
                    preparation: None,
                }
            } else {
                TypedFailure::Publication(retained)
            };
            Err((cause, owner))
        }
    }
}
pub(super) fn control_bytes<T: safemlx::ArrayElement + Send + 'static>() -> Option<usize> {
    let owner = Attachment::layout();
    let parts = [
        usize::try_from(
            OriginalHostSourceBank::publication_control_bytes::<Copy<'static, T>>(0).ok()?,
        )
        .ok()?,
        owner.allocation_bytes()?,
        owner.prepared_bytes(),
        owner.preparation_control_bytes(),
        owner.preparation_failure_bytes(),
        owner.attachment_failure_bytes(),
        owner.original_attachment_control_bytes(),
        CompletedOwnedHostCopy::control_bytes()?,
        size_of::<Copy<'static, T>>(),
        size_of::<Result<Array, (GgufHostCopyCause, TypedFailure<T>)>>(),
        size_of::<Result<(), (CopyFailure<T>, Attachment)>>(),
        size_of::<Result<Attachment, CopyFailure<T>>>(),
        size_of::<Result<ImmutableSourceInspection<'static>, CopyFailure<T>>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}
