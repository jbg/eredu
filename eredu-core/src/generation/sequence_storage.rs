//! Neutral storage ownership; no reservation, byte allowance or native grant.

use super::{FinishReason, GenerationError, GenerationSequence};
use crate::BackendFailure;
use std::{fmt, iter::FusedIterator, ops::Deref, sync::Arc};

pub(super) mod sealed {
    use super::GenerationError;

    pub trait Storage {
        fn tokens(&self) -> &[u32];
        fn eos_token_ids(&self) -> &[u32];
        fn push(&mut self, token: u32) -> Result<(), GenerationError>;
    }
}

/// Storage modes accepted by the shared sequence algorithm.
///
/// This trait is sealed. Providers implement [`RetainedGenerationStorage`] and
/// transfer their actual owner through the retained adapter instead.
pub trait GenerationSequenceStorage: sealed::Storage {}

/// Legacy EOS and token vectors used by the default sequence.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LegacyGenerationStorage {
    pub(super) eos_token_ids: Vec<u32>,
    pub(super) tokens: Vec<u32>,
}
impl GenerationSequenceStorage for LegacyGenerationStorage {}
impl sealed::Storage for LegacyGenerationStorage {
    fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
    fn push(&mut self, token: u32) -> Result<(), GenerationError> {
        self.tokens.push(token);
        Ok(())
    }
}

/// Immutable committed-token storage whose owner also retains its custody.
///
/// The returned slice must remain unchanged for the owner's lifetime. This is
/// an ordinary storage interface, not evidence of funding or bounded execution.
pub trait GenerationTokenIdStorage: fmt::Debug + Send + Sync + 'static {
    /// Committed ids, excluding unused initialized capacity.
    fn token_ids(&self) -> &[u32];

    /// Immutable terminal visible text, when originally selected by its provider.
    /// This storage projection creates no source or funding authority.
    fn terminal_text(&self) -> Option<&str> {
        None
    }

    /// Releases this actual strong owner through its concrete provider.
    ///
    /// The default preserves ordinary Arc destruction. A provider protecting the
    /// Arc allocation itself must override this on its concrete payload, close
    /// every strong-owner exit through that same retirement operation, and keep
    /// Weak owners from escaping. For example, `Arc::into_inner` can move the
    /// last concrete payload out after deallocation when every strong owner
    /// uses it and no separate Weak owner keeps the allocation alive.
    ///
    /// This method must not panic. Core supplies no funding or completion proof.
    fn retire(self: Arc<Self>) {
        drop(self);
    }
}
impl GenerationTokenIdStorage for Vec<u32> {
    fn token_ids(&self) -> &[u32] {
        self
    }
}

/// Read-only owning committed-token result.
///
/// Moving or cloning this carrier preserves its actual storage owner. It offers
/// no mutable slice, managed Vec extraction or copy-on-write operation.
///
/// ```compile_fail
/// use eredu_core::GenerationTokenIds;
/// let mut ids = GenerationTokenIds::from(vec![1, 2]);
/// ids[0] = 3;
/// ```
pub struct GenerationTokenIds {
    // None exists only during owned retirement; no raw Arc is exported.
    owner: Option<Arc<dyn GenerationTokenIdStorage>>,
}
impl Clone for GenerationTokenIds {
    fn clone(&self) -> Self {
        Self::from_owner(Arc::clone(
            self.owner.as_ref().expect("live immutable token owner"),
        ))
    }
}
impl Drop for GenerationTokenIds {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.retire();
        }
    }
}
impl GenerationTokenIds {
    /// Retains an immutable terminal-text view of this same existing owner.
    /// No new Arc allocation or byte copy occurs. Borrow-only/legacy providers
    /// return None; this does not upgrade their originally selected output mode.
    pub fn terminal_text(&self) -> Option<GenerationText> {
        let owner = self.owner.as_ref().expect("live immutable token owner");
        owner.terminal_text()?;
        Some(GenerationText(Some(Arc::clone(owner))))
    }
    /// Coerces an existing immutable owner without allocating or copying tokens.
    ///
    /// An accounting provider must preserve its allocation through the concrete
    /// retirement operation as well as order payload before final custody.
    /// Existing external strong/Weak aliases remain that provider's obligation;
    /// this constructor does not confer or validate accounting authority.
    pub fn from_owner(owner: Arc<dyn GenerationTokenIdStorage>) -> Self {
        Self { owner: Some(owner) }
    }

    /// Borrows only the committed prefix.
    pub fn as_slice(&self) -> &[u32] {
        self.owner
            .as_ref()
            .expect("live immutable token owner")
            .token_ids()
    }
}
/// Immutable owning text projection of the same committed-token payload.
/// No mutable String/Vec/raw Arc or copy-on-write exit is available.
///
/// ```compile_fail
/// fn mutate(mut text: eredu_core::GenerationText) { text.push_str("x"); }
/// ```
pub struct GenerationText(Option<Arc<dyn GenerationTokenIdStorage>>);
impl GenerationText {
    /// Borrows the complete terminal UTF-8 view for this owner lifetime.
    pub fn as_str(&self) -> &str {
        self.0
            .as_ref()
            .expect("live text owner")
            .terminal_text()
            .expect("immutable terminal text provider")
    }
}
impl Clone for GenerationText {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live text owner"))))
    }
}
impl Drop for GenerationText {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}
impl fmt::Debug for GenerationText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}
impl AsRef<str> for GenerationText {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl From<Vec<u32>> for GenerationTokenIds {
    /// Legacy conversion allocates its immutable Arc header, but does not copy
    /// the Vec's token buffer. Managed providers instead use `from_owner`.
    fn from(tokens: Vec<u32>) -> Self {
        Self::from_owner(Arc::new(tokens))
    }
}
impl fmt::Debug for GenerationTokenIds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl PartialEq for GenerationTokenIds {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for GenerationTokenIds {}
impl AsRef<[u32]> for GenerationTokenIds {
    fn as_ref(&self) -> &[u32] {
        self.as_slice()
    }
}
impl Deref for GenerationTokenIds {
    type Target = [u32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<'a> IntoIterator for &'a GenerationTokenIds {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}
impl IntoIterator for GenerationTokenIds {
    type Item = u32;
    type IntoIter = GenerationTokenIdsIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        let back = self.len();
        GenerationTokenIdsIntoIter {
            owner: self,
            front: 0,
            back,
        }
    }
}

/// Owning iterator which retains the result even after partial consumption.
#[derive(Debug)]
pub struct GenerationTokenIdsIntoIter {
    owner: GenerationTokenIds,
    front: usize,
    back: usize,
}
impl Iterator for GenerationTokenIdsIntoIter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.front == self.back {
            return None;
        }
        let token = self.owner[self.front];
        self.front += 1;
        Some(token)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}
impl DoubleEndedIterator for GenerationTokenIdsIntoIter {
    fn next_back(&mut self) -> Option<u32> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(self.owner[self.back])
    }
}
impl ExactSizeIterator for GenerationTokenIdsIntoIter {}
impl FusedIterator for GenerationTokenIdsIntoIter {}

/// Provider-owned fixed token slots and EOS policy, with its original custody.
///
/// Constructors and funding live with the provider. Core receives an already
/// owned Box; it neither allocates that Box nor accepts bytes plus a host guard.
/// The provider must price the concrete boxed representation, Arc allocation,
/// slots and error/control moves at its original admission boundary.
///
/// EOS ids are sorted, unique and immutable. After successful preparation, the
/// initialized token slice has exactly `max_tokens()` entries and never moves
/// or changes length. No immutable token owner may escape while slots are
/// mutable. Preparation owns partial allocations on error/unwind. Freeze moves
/// the existing Arc, including its committed length and final custody, without
/// allocating or copying a terminal buffer. These are storage obligations, not
/// a core funding or native completion certificate.
pub trait RetainedGenerationStorage: fmt::Debug + Send + Sync + 'static {
    /// Optional immutable concrete copy source. It grants no destination or native authority.
    fn copy_source(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Exact consumer descriptor consumed by this provider's original bank.
    /// The default has no consumer contribution. Returning a requested value
    /// without original binding/accounting does not satisfy this contract.
    fn consumer_layout(&self) -> Option<&crate::GenerationSequenceConsumerLayout> {
        None
    }
    /// Checks the exact synchronous source-bearing input returned by admission.
    /// No-input is the default; this does not certify accounting or compilation.
    fn matches_decoder_input(&self, input: Option<&dyn crate::GenerationDecoderInput>) -> bool {
        input.is_none()
    }
    /// Explicit output mode actually consumed by the original provider.
    fn decoder_output(&self) -> crate::GenerationDecoderOutput {
        crate::GenerationDecoderOutput::Suffix
    }
    /// Lends visible text from the same original decoder/stop owner.
    fn project_plain_text(
        &mut self,
        _id: u32,
    ) -> Result<crate::GenerationPlainText<'_>, crate::GenerationDecoderError> {
        Err(crate::GenerationDecoderError::Unavailable)
    }
    /// Checks decoder finish before lending the original pending stop text.
    fn finish_plain_text(&mut self) -> Result<&str, crate::GenerationDecoderError> {
        Err(crate::GenerationDecoderError::Unavailable)
    }
    /// Lends a decoded suffix from the original provider's own fixed storage.
    /// The caller must end the loan before committing the same token. No copy,
    /// second decoder, model submission or readiness vote occurs here.
    fn decode_token(&mut self, _id: u32) -> Result<Option<&str>, crate::GenerationDecoderError> {
        Err(crate::GenerationDecoderError::Unavailable)
    }
    /// Performs the source's non-flushing finish check. Cancellation skips it.
    fn finish_decoder(&mut self) -> Result<(), crate::GenerationDecoderError> {
        Err(crate::GenerationDecoderError::Unavailable)
    }
    /// Original finite output allowance, unchanged throughout this owner.
    fn max_tokens(&self) -> usize;
    /// Sorted and deduplicated immutable EOS policy.
    fn eos_token_ids(&self) -> &[u32];
    /// Performs this owner's one local token-slot preparation attempt.
    ///
    /// Core does not run readiness agreements. The caller must hold the returned
    /// result/owner before the existing first readiness call and submit no model
    /// work until that agreement succeeds.
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure>;
    /// Initialized slots; may be empty before preparation.
    fn token_slots(&self) -> &[u32];
    /// The sole mutable loan of initialized slots after preparation.
    fn token_slots_mut(&mut self) -> &mut [u32];
    /// Consumes the mutable owner into its immutable committed prefix.
    ///
    /// `committed` is zero on cancellation before preparation. The provider
    /// moves its existing owner into `GenerationTokenIds::from_owner`; it must
    /// not allocate, shrink, clone the token buffer or discard original custody.
    /// The concrete method also owns Box retirement: a provider protecting the
    /// Box allocation must first move its concrete value through a separate
    /// owned unboxing function, then perform fallible work or release custody.
    /// Core cannot retire a Box after handing it to this consuming method.
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds;

    /// Releases the mutable provider, including dormant and partial failures.
    ///
    /// The default preserves ordinary Box destruction. A provider protecting
    /// the Box allocation must override this with concrete owned retirement,
    /// keeping custody until after that allocation has been deallocated. This
    /// method must not panic and supplies no funding or completion certificate.
    fn retire(self: Box<Self>) {
        drop(self);
    }
}

/// Closed, non-Clone owner of a retained storage provider.
///
/// Core installs this wrapper before invoking any provider callback. Borrowing
/// exposes only the immutable storage interface; dropping dispatches the
/// provider's owned retirement. No raw Box or mutable provider can be extracted.
///
/// ```compile_fail
/// use eredu_core::{RetainedGenerationStorage, RetainedGenerationStorageOwner};
/// fn raw(owner: RetainedGenerationStorageOwner) -> Box<dyn RetainedGenerationStorage> {
///     owner
/// }
/// ```
pub struct RetainedGenerationStorageOwner {
    owner: Option<Box<dyn RetainedGenerationStorage>>,
}
impl RetainedGenerationStorageOwner {
    /// Closes an already-owned provider before any callback or fallible work.
    pub fn new(owner: Box<dyn RetainedGenerationStorage>) -> Self {
        Self { owner: Some(owner) }
    }

    fn storage_mut(&mut self) -> &mut (dyn RetainedGenerationStorage + 'static) {
        self.owner.as_deref_mut().expect("live mutable token owner")
    }

    fn into_token_ids(mut self, committed: usize) -> GenerationTokenIds {
        // The consuming provider method takes over retirement before callbacks.
        self.owner
            .take()
            .expect("live mutable token owner")
            .into_token_ids(committed)
    }
}
impl Deref for RetainedGenerationStorageOwner {
    type Target = dyn RetainedGenerationStorage;
    fn deref(&self) -> &Self::Target {
        self.owner.as_deref().expect("live mutable token owner")
    }
}
impl fmt::Debug for RetainedGenerationStorageOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}
impl Drop for RetainedGenerationStorageOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.retire();
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Preparation {
    Dormant,
    Attempted,
    Ready,
}

/// Sized, non-Clone erasure adapter for provider-owned retained storage.
#[derive(Debug)]
pub struct RetainedGenerationSequenceStorage {
    // Immutable provider geometry, distinct from a shortened logical endpoint.
    token_capacity: usize,
    committed: usize,
    preparation: Preparation,
    // Keep the actual payload/custody last, including failure diagnostics.
    owner: RetainedGenerationStorageOwner,
}

/// Shared lifecycle over one non-Clone retained mutable owner.
///
/// ```compile_fail
/// use eredu_core::RetainedGenerationSequence;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<RetainedGenerationSequence>();
/// ```
pub type RetainedGenerationSequence = GenerationSequence<RetainedGenerationSequenceStorage>;

impl GenerationSequenceStorage for RetainedGenerationSequenceStorage {}
impl sealed::Storage for RetainedGenerationSequenceStorage {
    fn tokens(&self) -> &[u32] {
        self.owner
            .token_slots()
            .get(..self.committed)
            .expect("retained provider preserves the initialized committed prefix")
    }
    fn eos_token_ids(&self) -> &[u32] {
        self.owner.eos_token_ids()
    }
    fn push(&mut self, token: u32) -> Result<(), GenerationError> {
        if self.preparation != Preparation::Ready {
            return Err(GenerationError::StorageNotReady);
        }
        let slot = self
            .owner
            .storage_mut()
            .token_slots_mut()
            .get_mut(self.committed)
            .ok_or(GenerationError::InvalidStorage)?;
        *slot = token;
        self.committed += 1;
        Ok(())
    }
}

/// Invalid retained EOS policy, preserving the original supplied owner.
#[derive(Debug)]
pub struct RetainedSequenceConstructionError {
    owner: RetainedGenerationStorageOwner,
}
impl RetainedSequenceConstructionError {
    /// Recovers the exact unchanged provider inside its closed retirement owner.
    /// It may be inspected, dropped or passed to `from_retained_owner`.
    pub fn into_storage(self) -> RetainedGenerationStorageOwner {
        self.owner
    }
}
impl fmt::Display for RetainedSequenceConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("retained EOS policy must be sorted and unique")
    }
}
impl std::error::Error for RetainedSequenceConstructionError {}

/// Failed local preparation retaining both its cause and the sequence owner.
#[derive(Debug)]
pub struct RetainedSequencePreparationError {
    cause: BackendFailure,
    sequence: RetainedGenerationSequence,
}
impl RetainedSequencePreparationError {
    /// Original neutral preparation cause.
    pub fn cause(&self) -> &BackendFailure {
        &self.cause
    }
    /// Borrow the retained sequence; a failed attempt cannot be retried.
    pub fn sequence(&self) -> &RetainedGenerationSequence {
        &self.sequence
    }
    /// Retires the cause while its sequence is still owned, then recovers that
    /// same sequence. The cause cannot escape without the storage owner.
    pub fn into_sequence(self) -> RetainedGenerationSequence {
        let Self { cause, sequence } = self;
        drop(cause);
        sequence
    }
}
impl fmt::Display for RetainedSequencePreparationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "generation token storage preparation failed: {}",
            self.cause
        )
    }
}
impl std::error::Error for RetainedSequencePreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

impl RetainedGenerationSequence {
    /// Borrows the provider's immutable original consumer association, if any.
    /// This exposes no custody or allocation capability.
    pub fn consumer_layout(&self) -> Option<&crate::GenerationSequenceConsumerLayout> {
        self.storage.owner.consumer_layout()
    }

    /// Checks the provider against the same synchronous original decoder input.
    /// This read-only comparison is identity evidence, not an admission grant.
    pub fn matches_decoder_input(&self, input: Option<&dyn crate::GenerationDecoderInput>) -> bool {
        self.storage.owner.matches_decoder_input(input)
    }
    /// Actual original output kind, checked again by private consumer construction.
    pub fn decoder_output(&self) -> crate::GenerationDecoderOutput {
        self.storage.owner.decoder_output()
    }
    /// Combined decoder/stop operation; one loan excludes another step or commit.
    pub fn project_plain_text(
        &mut self,
        id: u32,
    ) -> Result<crate::GenerationPlainText<'_>, crate::GenerationDecoderError> {
        if self.storage.preparation != Preparation::Ready {
            return Err(crate::GenerationDecoderError::Unprepared);
        }
        self.storage.owner.storage_mut().project_plain_text(id)
    }
    /// Performs non-flushing decoder validation then lends the stop tail.
    pub fn finish_plain_text(&mut self) -> Result<&str, crate::GenerationDecoderError> {
        if self.storage.preparation != Preparation::Ready && self.max_tokens != 0 {
            return Err(crate::GenerationDecoderError::Unprepared);
        }
        self.storage.owner.storage_mut().finish_plain_text()
    }
    /// Borrows the original decoder suffix until the next mutation/commit.
    pub fn decode_token(&mut self, id: u32) -> Result<Option<&str>, crate::GenerationDecoderError> {
        if self.storage.preparation != Preparation::Ready {
            return Err(crate::GenerationDecoderError::Unprepared);
        }
        self.storage.owner.storage_mut().decode_token(id)
    }
    /// Runs the admitted decoder's ordinary finish check. Empty original output
    /// may finish without allocating either token or decoder destinations.
    pub fn finish_decoder(&mut self) -> Result<(), crate::GenerationDecoderError> {
        if self.storage.preparation != Preparation::Ready && self.max_tokens != 0 {
            return Err(crate::GenerationDecoderError::Unprepared);
        }
        self.storage.owner.storage_mut().finish_decoder()
    }

    /// Borrows the policy and moves an already-owned provider into the adapter.
    /// No token/EOS/Box/Arc allocation or readiness agreement occurs here.
    pub fn from_retained_storage(
        owner: Box<dyn RetainedGenerationStorage>,
    ) -> Result<Self, RetainedSequenceConstructionError> {
        Self::from_retained_owner(RetainedGenerationStorageOwner::new(owner))
    }

    /// Reuses an already closed owner without exposing its Box. Policy checks
    /// and every provider callback run while that retirement wrapper is held.
    pub fn from_retained_owner(
        owner: RetainedGenerationStorageOwner,
    ) -> Result<Self, RetainedSequenceConstructionError> {
        if owner.eos_token_ids().windows(2).any(|w| w[0] >= w[1]) {
            return Err(RetainedSequenceConstructionError { owner });
        }
        let max_tokens = owner.max_tokens();
        Ok(Self {
            max_tokens,
            storage: RetainedGenerationSequenceStorage {
                token_capacity: max_tokens,
                committed: 0,
                preparation: Preparation::Dormant,
                owner,
            },
            finish_reason: (max_tokens == 0).then_some(FinishReason::MaxTokens),
        })
    }

    /// Attempts dormant materialization locally, preserving ownership on error.
    ///
    /// Store this result before the existing first readiness agreement. This
    /// method issues no vote, prediction, reservation or native permission.
    /// A ready sequence is returned unchanged; a failed attempt is not repeated.
    pub fn prepare_storage(mut self) -> Result<Self, RetainedSequencePreparationError> {
        let result = if self.is_finished() {
            Err(BackendFailure::from_error(GenerationError::AlreadyFinished))
        } else {
            match self.storage.preparation {
                Preparation::Ready => return Ok(self),
                Preparation::Attempted => {
                    Err(BackendFailure::from_error(GenerationError::StorageNotReady))
                }
                Preparation::Dormant => {
                    self.storage.preparation = Preparation::Attempted;
                    self.storage
                        .owner
                        .storage_mut()
                        .prepare_tokens()
                        .and_then(|()| {
                            if self.max_tokens > self.storage.token_capacity
                                || self.storage.owner.max_tokens() != self.storage.token_capacity
                                || self.storage.owner.token_slots().len()
                                    != self.storage.token_capacity
                                || self
                                    .storage
                                    .owner
                                    .eos_token_ids()
                                    .windows(2)
                                    .any(|w| w[0] >= w[1])
                            {
                                Err(BackendFailure::from_error(GenerationError::InvalidStorage))
                            } else {
                                Ok(())
                            }
                        })
                }
            }
        };
        match result {
            Ok(()) => {
                self.storage.preparation = Preparation::Ready;
                Ok(self)
            }
            Err(cause) => Err(RetainedSequencePreparationError {
                cause,
                sequence: self,
            }),
        }
    }

    pub(crate) fn matches_preparation(&self, max_tokens: usize, eos: &[u32]) -> bool {
        let stored = self.storage.owner.eos_token_ids();
        self.max_tokens == max_tokens
            && self.storage.committed == 0
            && self.storage.preparation == Preparation::Dormant
            && self.finish_reason == (max_tokens == 0).then_some(super::FinishReason::MaxTokens)
            && eos.iter().all(|id| stored.binary_search(id).is_ok())
            && stored.iter().all(|id| eos.contains(id))
    }

    /// Freezes the existing committed prefix by consuming its mutable owner.
    /// This also permits an empty result after cancellation without materializing
    /// token slots; no mutable owner remains after this transition.
    pub fn into_token_ids(self) -> GenerationTokenIds {
        self.storage.owner.into_token_ids(self.storage.committed)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod retirement_tests;

/// Borrowed lifecycle plan. The concrete runtime copies and admits its provider;
/// core retains the exact committed, readiness and finish state for reassembly.
pub struct RetainedGenerationSequenceCopy<'a> {
    source: &'a RetainedGenerationSequence,
}
impl RetainedGenerationSequence {
    /// Excludes source mutation throughout the complete provider copy transaction.
    pub fn prepare_host_copy(&self) -> RetainedGenerationSequenceCopy<'_> {
        RetainedGenerationSequenceCopy { source: self }
    }
}
impl<'a> RetainedGenerationSequenceCopy<'a> {
    /// Immutable provider loan for the owning runtime mechanism's checked downcast.
    pub fn provider(&self) -> &'a dyn RetainedGenerationStorage {
        &*self.source.storage.owner
    }
    /// Installs the independently admitted provider with unchanged source state.
    /// Provider source identity is the concrete copier's obligation; all public
    /// sequence policy and initialized bytes are checked before accepting it.
    pub fn complete(
        self,
        owner: RetainedGenerationStorageOwner,
    ) -> Result<RetainedGenerationSequence, RetainedSequenceCopyMismatch> {
        let old = &self.source.storage.owner;
        if owner.max_tokens() != old.max_tokens()
            || owner.eos_token_ids() != old.eos_token_ids()
            || owner.token_slots() != old.token_slots()
            || owner.consumer_layout() != old.consumer_layout()
            || owner.decoder_output() != old.decoder_output()
        {
            return Err(RetainedSequenceCopyMismatch { _owner: owner });
        }
        Ok(GenerationSequence {
            max_tokens: self.source.max_tokens,
            finish_reason: self.source.finish_reason,
            storage: RetainedGenerationSequenceStorage {
                token_capacity: self.source.storage.token_capacity,
                committed: self.source.storage.committed,
                preparation: self.source.storage.preparation,
                owner,
            },
        })
    }
}
/// A rejected copied provider retains its own account through destruction.
#[derive(Debug)]
pub struct RetainedSequenceCopyMismatch {
    _owner: RetainedGenerationStorageOwner,
}
impl fmt::Display for RetainedSequenceCopyMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("copied generation provider differs from its source")
    }
}
impl std::error::Error for RetainedSequenceCopyMismatch {}
