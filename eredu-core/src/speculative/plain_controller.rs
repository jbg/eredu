//! Actual plain-controller history and allocation-free decision sources.
use super::{SpeculativeBuffer, SpeculativeBufferAllocationError};
use crate::{HostPreparationAuthority, SharedTokenFilter, TextControllerStorage};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

/// Refusal before publishing a provisional plain controller.
#[derive(Debug, thiserror::Error)]
pub enum PlainControllerError {
    /// Opaque callbacks or storage have no fixed producer.
    #[error("controller has no prepared plain-state producer")]
    Unknown,
    /// Checked host geometry overflowed.
    #[error("plain controller host geometry overflow")]
    Overflow,
    /// The proposed suffix does not extend the durable prefix.
    #[error("constrained sampler history diverges from its committed logical prefix")]
    History,
    /// The actual tokenizer domain excludes this canonical ID.
    #[error("token {0} has no consistent tokenizer mapping")]
    InvalidToken(u32),
    /// Source, destination or copy capacity changed.
    #[error("plain controller source or copy geometry changed")]
    Source,
    /// Mutation requires an independently prepared destination with spare space.
    #[error("plain controller mutation requires a unique prepared history destination")]
    Destination,
    /// Real host allocation failure retains destination custody.
    #[error(transparent)]
    Allocation(#[from] SpeculativeBufferAllocationError),
}
struct Payload {
    values: SpeculativeBuffer<u32>,
}
struct Retained(Option<Arc<Payload>>);
impl Clone for Retained {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live history"))))
    }
}
impl Drop for Retained {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
enum Storage {
    Ordinary(Vec<u32>),
    Retained(Retained),
}
/// Canonical prefix. Ordinary state keeps Vec behavior. Retained aliases are
/// immutable until independently copied; retained mutation never allocates.
pub struct PlainControllerHistory(Storage);
impl std::fmt::Debug for PlainControllerHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}
impl Default for PlainControllerHistory {
    fn default() -> Self {
        Self(Storage::Ordinary(Vec::new()))
    }
}
impl Clone for PlainControllerHistory {
    fn clone(&self) -> Self {
        Self(match &self.0 {
            Storage::Ordinary(v) => Storage::Ordinary(v.clone()),
            Storage::Retained(v) => Storage::Retained(v.clone()),
        })
    }
}
impl Deref for PlainControllerHistory {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        match &self.0 {
            Storage::Ordinary(v) => v,
            Storage::Retained(v) => &v.0.as_ref().expect("live history").values,
        }
    }
}
impl PlainControllerHistory {
    pub(super) fn is_prepared(&self) -> bool {
        matches!(self.0, Storage::Retained(_))
    }
    /// Actual capacity; this does not permit growth.
    pub fn capacity(&self) -> usize {
        match &self.0 {
            Storage::Ordinary(v) => v.capacity(),
            Storage::Retained(v) => v.0.as_ref().expect("live history").values.capacity(),
        }
    }
    pub(super) fn is_unique_prepared(&self) -> bool {
        matches!(&self.0,Storage::Retained(value) if Arc::strong_count(value.0.as_ref().expect("live history"))==1)
    }
    /// Actual destination buffer, shared shell and constructor controls.
    pub fn copy_metadata_bytes(capacity: usize) -> Option<usize> {
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            SpeculativeBuffer::<u32>::retained_control_bytes(capacity)?,
            shared,
            size_of::<Payload>(),
            size_of::<Storage>(),
            size_of::<Retained>(),
            size_of::<Self>(),
            size_of::<Result<Self, PlainControllerError>>(),
            size_of::<PlainControllerError>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Copies into already-paid capacity. The real buffer and every alias keep
    /// custody; no old ordinary allocation is adopted.
    pub fn copy_prepared(
        &self,
        capacity: usize,
        host: HostPreparationAuthority,
    ) -> Result<Self, PlainControllerError> {
        if host.is_unmanaged() || capacity < self.len() {
            return Err(PlainControllerError::Source);
        }
        let mut values = SpeculativeBuffer::try_new_retained(capacity, host)?;
        values
            .try_extend(self.iter().copied())
            .map_err(|_| PlainControllerError::Destination)?;
        Ok(Self(Storage::Retained(Retained(Some(Arc::new(Payload {
            values,
        }))))))
    }
    /// Ordinary append, or nonallocating append to a unique prepared destination.
    pub fn try_push(&mut self, token: u32) -> Result<(), PlainControllerError> {
        match &mut self.0 {
            Storage::Ordinary(v) => {
                v.push(token);
                Ok(())
            }
            Storage::Retained(v) => Arc::get_mut(v.0.as_mut().expect("live history"))
                .ok_or(PlainControllerError::Destination)?
                .values
                .try_push(token)
                .map_err(|_| PlainControllerError::Destination),
        }
    }
    /// Prepared commitment cannot grow an ordinary destination.
    pub fn try_push_prepared(&mut self, token: u32) -> Result<(), PlainControllerError> {
        if matches!(self.0, Storage::Ordinary(_)) {
            return Err(PlainControllerError::Destination);
        }
        self.try_push(token)
    }
}

/// Borrowed fixed decision state, with its actual shared/original source
/// evidence. Native consumers must authenticate that evidence separately.
#[derive(Debug, Clone, Copy)]
pub struct PlainControllerSource<'a> {
    history: &'a PlainControllerHistory,
    validity: &'a SharedTokenFilter,
    storage: TextControllerStorage<'a>,
}
impl<'a> PlainControllerSource<'a> {
    /// Declares the actual complete plain state. Grammars or callbacks with
    /// additional semantics must not supply this projection.
    pub fn new(
        history: &'a PlainControllerHistory,
        validity: &'a SharedTokenFilter,
        storage: TextControllerStorage<'a>,
    ) -> Self {
        Self {
            history,
            validity,
            storage,
        }
    }
    /// Durable prefix, without a clone.
    pub fn history(self) -> &'a [u32] {
        self.history
    }
    /// Exact current capacity.
    pub fn capacity(self) -> usize {
        self.history.capacity()
    }
    /// Retained tokenizer-validity source.
    pub fn validity(self) -> &'a SharedTokenFilter {
        self.validity
    }
    /// Actual source evidence; no adoption or discount is implied.
    pub fn storage(self) -> TextControllerStorage<'a> {
        self.storage
    }
    /// Same canonical-domain predicate used by ordinary plain validation.
    pub fn validate_token(self, token: u32) -> Result<(), PlainControllerError> {
        if self.validity.allows(token) {
            Ok(())
        } else {
            Err(PlainControllerError::InvalidToken(token))
        }
    }
    /// Durable-prefix validation precedes suffix validation and its first error.
    pub fn validate_history(self, history: &[u32]) -> Result<(), PlainControllerError> {
        if !history.starts_with(self.history) {
            return Err(PlainControllerError::History);
        }
        for &token in &history[self.history.len()..] {
            self.validate_token(token)?;
        }
        Ok(())
    }
    /// Verifies exact immutable source ownership, prefix and copied capacity.
    pub fn matches_copy(self, copied: PlainControllerSource<'_>, capacity: usize) -> bool {
        self.validity.same_storage(copied.validity)
            && self.history() == copied.history()
            && copied.capacity() == capacity
            && copied.history.is_unique_prepared()
    }
}

/// Closed alias of one prepared prefix and its immutable filter. This is an
/// identity/lifetime witness only, never numerical or memory authority.
#[derive(Debug, Clone)]
pub struct PreparedPlainControllerIdentity {
    history: PlainControllerHistory,
    validity: SharedTokenFilter,
}
impl PreparedPlainControllerIdentity {
    /// Exact source ownership, without content-based substitution.
    pub fn same_source(&self, other: &Self) -> bool {
        self.history.same_prepared_source(&other.history)
            && self.validity.same_storage(&other.validity)
    }
    /// Fixed alias/transport controls. No new shared allocation is constructed.
    pub fn metadata_bytes() -> usize {
        size_of::<Self>()
            + size_of::<Result<Self, PlainControllerError>>()
            + size_of::<PlainControllerHistory>()
            + size_of::<SharedTokenFilter>()
    }
}
impl PlainControllerHistory {
    pub(super) fn same_prepared_source(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Storage::Retained(a), Storage::Retained(b)) => Arc::ptr_eq(
                a.0.as_ref().expect("live history"),
                b.0.as_ref().expect("live history"),
            ),
            _ => false,
        }
    }
}
impl PlainControllerSource<'_> {
    /// Retains the exact already-prepared source without copying its buffer.
    /// Ordinary histories refuse rather than laundering their Clone allocation.
    pub fn retain_prepared_identity(
        self,
    ) -> Result<PreparedPlainControllerIdentity, PlainControllerError> {
        if !matches!(self.history.0, Storage::Retained(_)) {
            return Err(PlainControllerError::Source);
        }
        Ok(PreparedPlainControllerIdentity {
            history: self.history.clone(),
            validity: self.validity.clone(),
        })
    }
    /// Checks an immutable completed-value witness against this actual prefix.
    pub fn matches_prepared_identity(self, identity: &PreparedPlainControllerIdentity) -> bool {
        self.history.same_prepared_source(&identity.history)
            && self.validity.same_storage(&identity.validity)
    }
}
