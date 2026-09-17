//! One concrete source allocation, closed through owned erasure and retirement.
use super::{
    BackendFailure, Error, GenerationSequenceBankRejection, HostMetadataFundingError,
    PreparedRequestRejection, TokenInputRejection,
};
use std::{fmt, mem::size_of, sync::Arc};

type ErrorObject = dyn Error + Send + Sync + 'static;

// Only this blanket implementation can enter SourceOwner. No raw erased owner,
// custom disposer or mutable source borrow is available to callers.
trait ErasedErrorSource: Send + Sync {
    fn error(&self) -> &ErrorObject;
    fn retire(self: Box<Self>);
    fn retire_shared(self: Arc<Self>);
}
impl<E: Error + Send + Sync + 'static> ErasedErrorSource for E {
    fn error(&self) -> &ErrorObject {
        self
    }
    fn retire(self: Box<Self>) {
        let value = unbox(self);
        #[cfg(test)]
        RETIRED.with(|count| count.set(count.get() + 1));
        drop(value);
    }
    fn retire_shared(self: Arc<Self>) {
        // This population exposes neither Arc nor Weak. The final strong exit
        // removes the allocation before dropping E, even across threads.
        let value = Arc::into_inner(self);
        #[cfg(test)]
        if value.is_some() {
            SHARED_RETIRED.with(|count| count.set(count.get() + 1));
        }
        drop(value);
    }
}

// The return boundary places disposal of E after disposal of its Box allocation.
// This helper runs no callback and does not depend on an enclosing local's Drop.
pub(super) fn unbox<E>(owner: Box<E>) -> E {
    *owner
}

pub(super) struct SourceOwner(OwnerKind);
enum OwnerKind {
    HostMetadata(HostMetadataFundingError),
    TextContext(&'static crate::backend::TextContextError),
    Owned(Option<Box<dyn ErasedErrorSource>>),
    Shared(Option<Arc<dyn ErasedErrorSource>>),
    // Only the corresponding fixed typed conversion constructs each static branch.
    SequenceBank(&'static GenerationSequenceBankRejection),
    TokenInput(&'static TokenInputRejection),
    PreparedControl(&'static crate::PreparedControlInputError),
    PreparedRequest(&'static PreparedRequestRejection),
}
impl SourceOwner {
    pub(super) fn metadata_funding(cause: HostMetadataFundingError) -> Self {
        Self(OwnerKind::HostMetadata(cause))
    }
    pub(super) fn shared<E: Error + Send + Sync + 'static>(source: E) -> Self {
        Self(OwnerKind::Shared(Some(Arc::new(source))))
    }
    pub(super) fn same_shared(&self, other: &Self) -> bool {
        let (OwnerKind::Shared(left), OwnerKind::Shared(right)) = (&self.0, &other.0) else {
            unreachable!("closed shared source comparison")
        };
        Arc::ptr_eq(left.as_ref().expect("live source"), right.as_ref().expect("live source"))
    }
    pub(super) fn retain_shared(&self) -> Self {
        let OwnerKind::Shared(source) = &self.0 else {
            unreachable!("closed shared source")
        };
        Self(OwnerKind::Shared(Some(
            source.as_ref().expect("live shared source").clone(),
        )))
    }
    pub(super) fn text_context(cause: crate::backend::TextContextError) -> Self {
        use crate::backend::TextContextError as E;
        static RUN: E = E::RunExhausted;
        static POLICY: E = E::PolicyExhausted;
        Self(OwnerKind::TextContext(match cause {
            E::RunExhausted => &RUN,
            E::PolicyExhausted => &POLICY,
        }))
    }

    pub(super) fn new<E: Error + Send + Sync + 'static>(source: E) -> Self {
        Self::from_box(Box::new(source))
    }
    pub(super) fn from_box<E: Error + Send + Sync + 'static>(source: Box<E>) -> Self {
        Self(OwnerKind::Owned(Some(source)))
    }
    pub(super) fn sequence_bank_rejection(cause: GenerationSequenceBankRejection) -> Self {
        static UNAVAILABLE: GenerationSequenceBankRejection =
            GenerationSequenceBankRejection::Unavailable;
        static BUSY: GenerationSequenceBankRejection = GenerationSequenceBankRejection::Busy;
        static IDENTITY: GenerationSequenceBankRejection =
            GenerationSequenceBankRejection::IdentityMismatch;
        Self(OwnerKind::SequenceBank(match cause {
            GenerationSequenceBankRejection::Unavailable => &UNAVAILABLE,
            GenerationSequenceBankRejection::Busy => &BUSY,
            GenerationSequenceBankRejection::IdentityMismatch => &IDENTITY,
        }))
    }
    pub(super) fn token_input_rejection(cause: TokenInputRejection) -> Self {
        // Only these seven closed typed diagnostics can use this branch.
        static VALUES: [TokenInputRejection; 7] = [
            TokenInputRejection::Unavailable,
            TokenInputRejection::Busy,
            TokenInputRejection::IdentityMismatch,
            TokenInputRejection::Unsupported,
            TokenInputRejection::Empty,
            TokenInputRejection::Overflow,
            TokenInputRejection::InvalidToken,
        ];
        let index = match cause {
            TokenInputRejection::Unavailable => 0,
            TokenInputRejection::Busy => 1,
            TokenInputRejection::IdentityMismatch => 2,
            TokenInputRejection::Unsupported => 3,
            TokenInputRejection::Empty => 4,
            TokenInputRejection::Overflow => 5,
            TokenInputRejection::InvalidToken => 6,
        };
        Self(OwnerKind::TokenInput(&VALUES[index]))
    }
    pub(super) fn prepared_control_rejection(cause: crate::PreparedControlInputError) -> Self {
        use crate::PreparedControlInputError as E;
        static VALUES: [E; 8] = [
            E::UnknownBound,
            E::InstrumentationUnavailable,
            E::Unsupported,
            E::MissingSourceIdentity,
            E::UnknownFrontier,
            E::SourceMismatch,
            E::InvalidAttribution,
            E::Overflow,
        ];
        let index = match cause {
            E::UnknownBound => 0,
            E::InstrumentationUnavailable => 1,
            E::Unsupported => 2,
            E::MissingSourceIdentity => 3,
            E::UnknownFrontier => 4,
            E::SourceMismatch => 5,
            E::InvalidAttribution => 6,
            E::Overflow => 7,
        };
        Self(OwnerKind::PreparedControl(&VALUES[index]))
    }
    pub(super) fn prepared_request_rejection(cause: PreparedRequestRejection) -> Self {
        use PreparedRequestRejection as E;
        static VALUES: [E; 13] = [
            E::Unsupported,
            E::MissingSequence,
            E::SourceUnavailable,
            E::IdentityMismatch,
            E::RequestMismatch,
            E::Busy,
            E::MissingInspectionStorage,
            E::MissingNumericalWorkspace,
            E::MissingExecutionControls,
            E::MissingCapture,
            E::Overflow,
            E::CapacityExceeded,
            E::MissingController,
        ];
        let index = match cause {
            E::Unsupported => 0,
            E::MissingSequence => 1,
            E::SourceUnavailable => 2,
            E::IdentityMismatch => 3,
            E::RequestMismatch => 4,
            E::Busy => 5,
            E::MissingInspectionStorage => 6,
            E::MissingNumericalWorkspace => 7,
            E::MissingExecutionControls => 8,
            E::MissingCapture => 9,
            E::Overflow => 10,
            E::CapacityExceeded => 11,
            E::MissingController => 12,
        };
        Self(OwnerKind::PreparedRequest(&VALUES[index]))
    }
    pub(super) fn error(&self) -> &ErrorObject {
        match &self.0 {
            OwnerKind::HostMetadata(cause) => cause,
            OwnerKind::TextContext(cause) => *cause,
            OwnerKind::Owned(owner) => owner.as_deref().expect("live backend error source").error(),
            OwnerKind::Shared(owner) => owner.as_deref().expect("live shared error source").error(),
            OwnerKind::SequenceBank(cause) => *cause,
            OwnerKind::TokenInput(cause) => *cause,
            OwnerKind::PreparedControl(cause) => *cause,
            OwnerKind::PreparedRequest(cause) => *cause,
        }
    }
}
impl fmt::Debug for SourceOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.error(), f)
    }
}
impl fmt::Display for SourceOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.error(), f)
    }
}
impl Drop for SourceOwner {
    fn drop(&mut self) {
        match &mut self.0 {
            OwnerKind::Owned(owner) => {
                if let Some(source) = owner.take() {
                    source.retire();
                }
            }
            OwnerKind::Shared(owner) => {
                if let Some(source) = owner.take() {
                    source.retire_shared();
                }
            }
            _ => {}
        }
    }
}

// Requested allocation payload and conservative named finite value overlap.
// Nested allocations inside E, caller wrapping and formatting are not included.
pub(super) fn retention_peak_bytes<E: Error + Send + Sync + 'static>() -> Option<usize> {
    let parts = [
        size_of::<E>(),         // one source Box allocation, with E's actual padding
        size_of::<E>(),         // caller/public constructor argument
        size_of::<Option<E>>(), // safe movable Any check
        size_of::<Option<BackendFailure>>(), // neutral transfer branch
        size_of::<&mut dyn std::any::Any>(),
        size_of::<Option<&mut Option<HostMetadataFundingError>>>(),
        size_of::<E>(), // generic helper or concrete retirement return value
        size_of::<BackendFailure>(), // returned or flattening failure value
        size_of::<SourceOwner>(), // erased owner under construction/retirement
        size_of::<Option<Box<dyn ErasedErrorSource>>>(), // taken owner slot
        size_of::<Box<E>>(), // concrete allocation/unbox argument
        size_of::<Box<dyn ErasedErrorSource>>(), // consumed retirement dispatch
        size_of::<Box<ErrorObject>>(), // from_error's same-allocation type check
        size_of::<Result<Box<BackendFailure>, Box<ErrorObject>>>(), // flattening branch
        size_of::<Result<Box<E>, Box<ErrorObject>>>(), // exact generic recapture
    ];
    parts.into_iter().try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
thread_local! {
    // A safe semantic boundary witness, not a native allocator observation.
    static RETIRED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static SHARED_RETIRED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(super) fn retirement_count() -> usize {
    RETIRED.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn shared_retirement_count() -> usize {
    SHARED_RETIRED.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests;

// Pinned Rust1.98 ArcInner is repr(C,align(2)): two integer atomic counters,
// then E. Nested payloads, caller containers and allocator overhead are separate.
pub(super) fn shared_retention_peak_bytes<E: Error + Send + Sync + 'static>() -> Option<usize> {
    use std::{
        alloc::Layout,
        mem::ManuallyDrop,
        sync::{Weak, atomic::AtomicUsize},
    };
    let allocation = Layout::new::<[AtomicUsize; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<E>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    [
        size_of::<E>(),
        size_of::<E>(),
        size_of::<Option<E>>(),
        size_of::<BackendFailure>(),
        size_of::<super::SharedBackendFailure>(),
        size_of::<Option<BackendFailure>>(),
        size_of::<&mut dyn std::any::Any>(),
        size_of::<SourceOwner>(),
        size_of::<Arc<E>>(),
        size_of::<Weak<E>>(),
        size_of::<ManuallyDrop<Arc<E>>>(),
        size_of::<Option<Arc<dyn ErasedErrorSource>>>(),
        size_of::<Arc<dyn ErasedErrorSource>>(),
    ]
    .into_iter()
    .try_fold(allocation, usize::checked_add)
}
