//! Actual finite token/EOS storage for the shared sequence algorithm.
use crate::{
    BackendFailure, FinishReason, GenerationCancellationToken, GenerationError, GenerationSequence,
    GenerationTokenIdStorage, GenerationTokenIds, HostMetadataFundingError,
    HostPreparationAuthority, RetainedGenerationSequence, RetainedGenerationStorage,
    RetainedGenerationStorageOwner, RetainedSequenceConstructionError,
    RetainedSequenceCopyMismatch, RetainedSequencePreparationError, TokenCommit,
    TokenTerminalSignals,
};
use std::{
    alloc::Layout,
    ops::Deref,
    sync::{atomic::AtomicUsize, Arc},
};

/// Canonical sequence storage. Retained sequences have no infallible copy or
/// mutable/raw storage export; every branch uses the same GenerationSequence rules.
#[derive(Debug)]
pub enum SpeculativeSequence {
    /// Existing ordinary storage and allocation behavior.
    Ordinary(GenerationSequence),
    /// A concrete provider retains its storage custody through terminal output.
    Retained(RetainedGenerationSequence),
}
impl From<GenerationSequence> for SpeculativeSequence {
    fn from(value: GenerationSequence) -> Self {
        Self::Ordinary(value)
    }
}
impl From<RetainedGenerationSequence> for SpeculativeSequence {
    fn from(value: RetainedGenerationSequence) -> Self {
        Self::Retained(value)
    }
}
/// Immutable copy source; borrowing alone grants no storage or execution authority.
#[derive(Clone, Copy, Debug)]
pub enum SpeculativeSequenceRef<'a> {
    /// Borrow of the existing ordinary sequence.
    Ordinary(&'a GenerationSequence),
    /// Borrow of a retained provider and its complete sequence state.
    Retained(&'a RetainedGenerationSequence),
}
impl<'a> From<&'a GenerationSequence> for SpeculativeSequenceRef<'a> {
    fn from(value: &'a GenerationSequence) -> Self {
        Self::Ordinary(value)
    }
}
impl<'a> From<&'a SpeculativeSequence> for SpeculativeSequenceRef<'a> {
    fn from(value: &'a SpeculativeSequence) -> Self {
        match value {
            SpeculativeSequence::Ordinary(s) => Self::Ordinary(s),
            SpeculativeSequence::Retained(s) => Self::Retained(s),
        }
    }
}
impl SpeculativeSequenceRef<'_> {
    /// Ordinary defaults never copy or relabel a retained provider.
    pub fn copy_ordinary(self) -> Result<SpeculativeSequence, BackendFailure> {
        match self {
            Self::Ordinary(s) => Ok(s.clone().into()),
            Self::Retained(_) => Err(HostMetadataFundingError::Unavailable.into()),
        }
    }
    /// Exact fixed-provider geometry. Unknown providers and ordinary sources do
    /// not become eligible merely because their names/shapes match.
    pub fn retained_geometry(self) -> Option<(usize, usize)> {
        let Self::Retained(s) = self else { return None };
        let source = s.prepare_host_copy();
        let provider = source
            .provider()
            .copy_source()?
            .downcast_ref::<Provider>()?;
        Some((provider.max_tokens(), provider.eos_token_ids().len()))
    }
    /// Creates an independent destination after its complete constructor query
    /// has been paid. All source slots and shared sequence state are preserved.
    pub fn try_copy_retained(
        self,
        authority: HostPreparationAuthority,
    ) -> Result<SpeculativeSequence, SpeculativeSequenceAllocationError> {
        let result = (|| {
            let Self::Retained(s) = self else {
                return Err(Cause::Source);
            };
            let copy = s.prepare_host_copy();
            let source = copy
                .provider()
                .copy_source()
                .and_then(|s| s.downcast_ref::<Provider>())
                .ok_or(Cause::Source)?;
            let owner = provider(
                source.max_tokens(),
                source.eos_token_ids(),
                Some(source.token_slots()),
                &authority,
            )?;
            copy.complete(owner)
                .map(SpeculativeSequence::Retained)
                .map_err(Cause::Copy)
        })();
        result.map_err(|cause| SpeculativeSequenceAllocationError { cause, authority })
    }
}
impl SpeculativeSequence {
    /// Allocates the actual fixed provider after the caller pays its query.
    /// EOS sorting/deduplication and termination remain the ordinary sequence rules.
    pub fn try_new_retained(
        maximum: usize,
        eos: &[u32],
        authority: HostPreparationAuthority,
    ) -> Result<Self, SpeculativeSequenceAllocationError> {
        let result = (|| {
            let owner = provider(maximum, eos, None, &authority)?;
            let sequence = RetainedGenerationSequence::from_retained_owner(owner)
                .map_err(Cause::Construction)?;
            // Empty output is already terminal; core deliberately does not run
            // first-token readiness for it. Its initialized empty provider is valid.
            if maximum == 0 {
                Ok(Self::Retained(sequence))
            } else {
                sequence
                    .prepare_storage()
                    .map(Self::Retained)
                    .map_err(Cause::Preparation)
            }
        })();
        result.map_err(|cause| SpeculativeSequenceAllocationError { cause, authority })
    }
    /// Requested buffers and concrete Arc/Box, constructor, copy, error and
    /// terminal controls. H construction/outer error transport remain caller-owned.
    pub fn retained_control_bytes(maximum: usize, eos: usize) -> Option<usize> {
        use std::mem::size_of;
        let arc = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        let parts = [
            Layout::array::<u32>(maximum).ok()?.size(),
            Layout::array::<u32>(eos).ok()?.size(),
            arc,
            size_of::<Provider>(),
            size_of::<Provider>(),
            size_of::<Payload>(),
            size_of::<Option<Payload>>(),
            size_of::<PayloadOwner>(),
            size_of::<Vec<u32>>() * 2,
            size_of::<Self>(),
            size_of::<RetainedGenerationSequence>(),
            size_of::<RetainedGenerationStorageOwner>(),
            size_of::<SpeculativeSequenceAllocationError>(),
            size_of::<Cause>(),
            size_of::<Result<Self, SpeculativeSequenceAllocationError>>(),
            size_of::<Result<RetainedGenerationStorageOwner, Cause>>(),
            size_of::<Result<RetainedGenerationSequence, RetainedSequenceConstructionError>>(),
            size_of::<Result<RetainedGenerationSequence, RetainedSequencePreparationError>>(),
            size_of::<Result<RetainedGenerationSequence, RetainedSequenceCopyMismatch>>(),
            size_of::<crate::RetainedGenerationSequenceCopy<'static>>(),
            size_of::<(usize, &[u32], Option<&[u32]>, &HostPreparationAuthority)>(),
            size_of::<GenerationTokenIds>(),
            size_of::<crate::GenerationTokenIdsIntoIter>(),
            size_of::<SpeculativeTokenIds>(),
            size_of::<SpeculativeTokenIdsIntoIter>(),
            size_of::<SpeculativeSequenceRef<'static>>(),
            BackendFailure::source_retention_peak_bytes::<GenerationError>()?,
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Committed prefix only.
    pub fn tokens(&self) -> &[u32] {
        match self {
            Self::Ordinary(s) => s.tokens(),
            Self::Retained(s) => s.tokens(),
        }
    }
    /// Remaining shared output allowance.
    pub fn remaining(&self) -> usize {
        match self {
            Self::Ordinary(s) => s.remaining(),
            Self::Retained(s) => s.remaining(),
        }
    }
    /// Original selected output cap.
    pub const fn max_tokens(&self) -> usize {
        match self {
            Self::Ordinary(s) => s.max_tokens(),
            Self::Retained(s) => s.max_tokens(),
        }
    }
    /// Shared terminal reason.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        match self {
            Self::Ordinary(s) => s.finish_reason(),
            Self::Retained(s) => s.finish_reason(),
        }
    }
    /// Whether the shared sequence is terminal.
    pub const fn is_finished(&self) -> bool {
        match self {
            Self::Ordinary(s) => s.is_finished(),
            Self::Retained(s) => s.is_finished(),
        }
    }
    /// Executes the existing commit/termination worker on the actual storage.
    pub fn commit(
        &mut self,
        token: u32,
        signals: TokenTerminalSignals,
    ) -> Result<TokenCommit, GenerationError> {
        match self {
            Self::Ordinary(s) => s.commit(token, signals),
            Self::Retained(s) => s.commit(token, signals),
        }
    }
    /// Executes shared cancellation.
    pub fn cancel(&mut self) -> bool {
        match self {
            Self::Ordinary(s) => s.cancel(),
            Self::Retained(s) => s.cancel(),
        }
    }
    /// Executes shared cancellation observation.
    pub fn observe_cancellation(&mut self, c: &GenerationCancellationToken) -> bool {
        match self {
            Self::Ordinary(s) => s.observe_cancellation(c),
            Self::Retained(s) => s.observe_cancellation(c),
        }
    }
    /// Exact fixed-provider copy construction request, or legacy logical bytes.
    /// Unknown retained providers keep the existing fail-closed estimate.
    pub fn snapshot_storage_bytes(&self) -> Option<u64> {
        match self {
            Self::Ordinary(s) => s.snapshot_storage_bytes(),
            Self::Retained(_) => {
                let (m, e) = SpeculativeSequenceRef::from(self).retained_geometry()?;
                u64::try_from(Self::retained_control_bytes(m, e)?).ok()
            }
        }
    }
    /// Consumes storage without copying retained tokens or creating an Arc.
    pub fn into_token_ids(self) -> SpeculativeTokenIds {
        match self {
            Self::Ordinary(s) => SpeculativeTokenIds::Ordinary(s.into_tokens()),
            Self::Retained(s) => SpeculativeTokenIds::Retained(s.into_token_ids()),
        }
    }
}

/// Terminal token custody. Only ordinary storage can yield a raw Vec.
#[derive(Debug)]
pub enum SpeculativeTokenIds {
    /// Existing ordinary output allocation.
    Ordinary(Vec<u32>),
    /// Existing immutable retained output owner.
    Retained(GenerationTokenIds),
}
impl Default for SpeculativeTokenIds {
    fn default() -> Self {
        Self::Ordinary(Vec::new())
    }
}
impl SpeculativeTokenIds {
    /// Consumes ordinary storage; retained storage is returned unchanged.
    pub fn try_into_ordinary(self) -> Result<Vec<u32>, Self> {
        match self {
            Self::Ordinary(v) => Ok(v),
            other => Err(other),
        }
    }
}
impl Deref for SpeculativeTokenIds {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        match self {
            Self::Ordinary(v) => v,
            Self::Retained(v) => v,
        }
    }
}
impl AsRef<[u32]> for SpeculativeTokenIds {
    fn as_ref(&self) -> &[u32] {
        self
    }
}
impl PartialEq for SpeculativeTokenIds {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for SpeculativeTokenIds {}
impl PartialEq<Vec<u32>> for SpeculativeTokenIds {
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.as_ref() == other
    }
}
impl<const N: usize> PartialEq<[u32; N]> for SpeculativeTokenIds {
    fn eq(&self, other: &[u32; N]) -> bool {
        self.as_ref() == other
    }
}

/// Consuming terminal iteration retains the actual owner through partial use.
#[derive(Debug)]
pub enum SpeculativeTokenIdsIntoIter {
    /// Ordinary owned token iterator.
    Ordinary(std::vec::IntoIter<u32>),
    /// Immutable retained token iterator.
    Retained(crate::GenerationTokenIdsIntoIter),
}
impl Iterator for SpeculativeTokenIdsIntoIter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        match self {
            Self::Ordinary(i) => i.next(),
            Self::Retained(i) => i.next(),
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Ordinary(i) => i.size_hint(),
            Self::Retained(i) => i.size_hint(),
        }
    }
}
impl DoubleEndedIterator for SpeculativeTokenIdsIntoIter {
    fn next_back(&mut self) -> Option<u32> {
        match self {
            Self::Ordinary(i) => i.next_back(),
            Self::Retained(i) => i.next_back(),
        }
    }
}
impl ExactSizeIterator for SpeculativeTokenIdsIntoIter {}
impl std::iter::FusedIterator for SpeculativeTokenIdsIntoIter {}
impl IntoIterator for SpeculativeTokenIds {
    type Item = u32;
    type IntoIter = SpeculativeTokenIdsIntoIter;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Ordinary(v) => SpeculativeTokenIdsIntoIter::Ordinary(v.into_iter()),
            Self::Retained(v) => SpeculativeTokenIdsIntoIter::Retained(v.into_iter()),
        }
    }
}
impl<'a> IntoIterator for &'a SpeculativeTokenIds {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("sequence destination allocation failed: {0}")]
    Reserve(#[source] std::collections::TryReserveError),
    #[error("sequence destination capacity differs from its request")]
    Capacity,
    #[error("sequence source is not this fixed retained provider")]
    Source,
    #[error("{0}")]
    Construction(#[source] RetainedSequenceConstructionError),
    #[error("{0}")]
    Preparation(#[source] RetainedSequencePreparationError),
    #[error("{0}")]
    Copy(#[source] RetainedSequenceCopyMismatch),
}
/// Exact allocation/refusal cause retains its constructor authority until last.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct SpeculativeSequenceAllocationError {
    #[source]
    cause: Cause,
    authority: HostPreparationAuthority,
}
#[derive(Debug)]
struct Payload {
    tokens: Vec<u32>,
    eos: Vec<u32>,
    committed: usize,
    _authority: HostPreparationAuthority,
}
impl GenerationTokenIdStorage for Payload {
    fn token_ids(&self) -> &[u32] {
        &self.tokens[..self.committed]
    }
    fn retire(self: Arc<Self>) {
        drop(Arc::into_inner(self));
    }
}
#[derive(Debug)]
struct PayloadOwner(Option<Arc<Payload>>);
impl PayloadOwner {
    fn get(&self) -> &Payload {
        self.0.as_deref().expect("live fixed sequence")
    }
    fn get_mut(&mut self) -> &mut Payload {
        Arc::get_mut(self.0.as_mut().expect("live fixed sequence")).expect("sole mutable sequence")
    }
    fn freeze(mut self, committed: usize) -> GenerationTokenIds {
        self.get_mut().committed = committed;
        GenerationTokenIds::from_owner(self.0.take().expect("live fixed sequence"))
    }
}
impl Drop for PayloadOwner {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            p.retire()
        }
    }
}
#[derive(Debug)]
struct Provider(PayloadOwner);
fn unbox<T>(owner: Box<T>) -> T {
    *owner
}
impl RetainedGenerationStorage for Provider {
    fn copy_source(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn max_tokens(&self) -> usize {
        self.0.get().tokens.len()
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.0.get().eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn token_slots(&self) -> &[u32] {
        &self.0.get().tokens
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut self.0.get_mut().tokens
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        unbox(self).0.freeze(committed)
    }
    fn retire(self: Box<Self>) {
        drop(unbox(self));
    }
}
fn vector(capacity: usize) -> Result<Vec<u32>, Cause> {
    let mut v = Vec::new();
    v.try_reserve_exact(capacity).map_err(Cause::Reserve)?;
    if v.capacity() != capacity {
        return Err(Cause::Capacity);
    }
    Ok(v)
}
fn provider(
    maximum: usize,
    eos: &[u32],
    source: Option<&[u32]>,
    authority: &HostPreparationAuthority,
) -> Result<RetainedGenerationStorageOwner, Cause> {
    let mut tokens = vector(maximum)?;
    if let Some(source) = source {
        if source.len() != maximum {
            return Err(Cause::Source);
        }
        tokens.extend_from_slice(source)
    } else {
        tokens.resize(maximum, 0)
    }
    let mut policy = vector(eos.len())?;
    policy.extend_from_slice(eos);
    policy.sort_unstable();
    policy.dedup();
    let payload = PayloadOwner(Some(Arc::new(Payload {
        tokens,
        eos: policy,
        committed: 0,
        _authority: authority.clone(),
    })));
    Ok(RetainedGenerationStorageOwner::new(Box::new(Provider(
        payload,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Debug)]
    struct Counted(Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn host(drops: &Arc<AtomicUsize>) -> HostPreparationAuthority {
        HostPreparationAuthority::retain(Counted(drops.clone()))
    }
    #[test]
    fn shared_sequence_copy_and_terminal_iteration_preserve_rules_and_custody() {
        let original_drops = Arc::new(AtomicUsize::new(0));
        let copy_drops = Arc::new(AtomicUsize::new(0));
        let mut ordinary = GenerationSequence::new(3, [9, 2, 9]);
        let mut retained =
            SpeculativeSequence::try_new_retained(3, &[9, 2, 9], host(&original_drops)).unwrap();
        let signals = TokenTerminalSignals {
            stop_sequence: false,
            grammar_complete: false,
        };
        assert_eq!(
            retained.commit(4, signals).unwrap(),
            ordinary.commit(4, signals).unwrap()
        );
        assert!(SpeculativeSequenceRef::from(&retained)
            .copy_ordinary()
            .is_err());
        let mut copied = SpeculativeSequenceRef::from(&retained)
            .try_copy_retained(host(&copy_drops))
            .unwrap();
        assert_eq!(
            copied.commit(2, signals).unwrap(),
            ordinary.clone().commit(2, signals).unwrap()
        );
        assert_eq!(copied.finish_reason(), Some(FinishReason::Eos));
        assert_eq!(retained.tokens(), [4]);
        assert_eq!(
            retained.commit(5, signals).unwrap(),
            ordinary.commit(5, signals).unwrap()
        );
        assert_eq!(
            retained.commit(6, signals).unwrap(),
            ordinary.commit(6, signals).unwrap()
        );
        assert_eq!(retained.finish_reason(), Some(FinishReason::MaxTokens));
        let terminal = retained.into_token_ids();
        let terminal = terminal.try_into_ordinary().unwrap_err();
        let mut ids = terminal.into_iter();
        assert_eq!(ids.next(), Some(4));
        assert_eq!(original_drops.load(Ordering::SeqCst), 0);
        drop(ids); // unused committed slots and the actual shared shell retire first
        assert_eq!(original_drops.load(Ordering::SeqCst), 1);
        assert_eq!(copied.tokens(), [4, 2]);
        let terminal = copied.into_token_ids();
        assert_eq!(copy_drops.load(Ordering::SeqCst), 0);
        drop(terminal);
        assert_eq!(copy_drops.load(Ordering::SeqCst), 1);
        let empty_drops = Arc::new(AtomicUsize::new(0));
        let empty = SpeculativeSequence::try_new_retained(0, &[], host(&empty_drops)).unwrap();
        assert_eq!(empty.finish_reason(), Some(FinishReason::MaxTokens));
        drop(empty.into_token_ids());
        assert_eq!(empty_drops.load(Ordering::SeqCst), 1);
    }
}
