use super::*;
use crate::{BackendFailureKind, GenerationCancellationToken, TokenTerminalSignals};
use std::{
    error::Error as _,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
struct Probe {
    prepared: AtomicUsize,
    retired: AtomicUsize,
    error_retired: AtomicUsize,
    error_without_owner: AtomicUsize,
}
#[derive(Debug)]
struct Custody(Arc<Probe>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.retired.fetch_add(1, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Payload {
    slots: Vec<u32>,
    eos: Vec<u32>,
    committed: usize,
    custody: Custody,
}
impl GenerationTokenIdStorage for Payload {
    fn token_ids(&self) -> &[u32] {
        &self.slots[..self.committed]
    }
}
#[derive(Debug, Clone, Copy)]
enum Behavior {
    Ready,
    Fail,
    WrongLength,
    Panic,
}
#[derive(Debug)]
struct Fixed {
    maximum: usize,
    behavior: Behavior,
    payload: Arc<Payload>,
}
#[derive(Debug)]
struct PreparationFailure(Arc<Probe>);
impl fmt::Display for PreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("fixture failed after retaining a partial slot allocation")
    }
}
impl std::error::Error for PreparationFailure {}
impl Drop for PreparationFailure {
    fn drop(&mut self) {
        self.0.error_retired.fetch_add(1, Ordering::SeqCst);
        if self.0.retired.load(Ordering::SeqCst) != 0 {
            self.0.error_without_owner.fetch_add(1, Ordering::SeqCst);
        }
    }
}
impl RetainedGenerationStorage for Fixed {
    fn max_tokens(&self) -> usize {
        self.maximum
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.payload.eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        let payload = Arc::get_mut(&mut self.payload).expect("sole mutable owner");
        payload.custody.0.prepared.fetch_add(1, Ordering::SeqCst);
        let extent = match self.behavior {
            Behavior::Ready => self.maximum,
            Behavior::Fail | Behavior::Panic => 1.min(self.maximum),
            Behavior::WrongLength => self.maximum + 1,
        };
        payload
            .slots
            .try_reserve_exact(extent)
            .map_err(|error| BackendFailure::new(BackendFailureKind::ResourceExhausted, error))?;
        payload.slots.resize(extent, 0);
        match self.behavior {
            Behavior::Ready | Behavior::WrongLength => Ok(()),
            Behavior::Fail => Err(BackendFailure::new(
                BackendFailureKind::ResourceExhausted,
                PreparationFailure(Arc::clone(&payload.custody.0)),
            )),
            Behavior::Panic => std::panic::panic_any("exact slot preparation panic"),
        }
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.slots
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut Arc::get_mut(&mut self.payload)
            .expect("sole mutable owner")
            .slots
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let Self { mut payload, .. } = *self;
        Arc::get_mut(&mut payload)
            .expect("sole owner at freeze")
            .committed = committed;
        GenerationTokenIds::from_owner(payload)
    }
}

fn storage(maximum: usize, eos: Vec<u32>, behavior: Behavior) -> (Box<Fixed>, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    (
        Box::new(Fixed {
            maximum,
            behavior,
            payload: Arc::new(Payload {
                slots: Vec::new(),
                eos,
                committed: 0,
                custody: Custody(Arc::clone(&probe)),
            }),
        }),
        probe,
    )
}
fn sequence(maximum: usize, eos: &[u32]) -> (RetainedGenerationSequence, Arc<Probe>) {
    let mut eos = eos.to_vec();
    eos.sort_unstable();
    eos.dedup();
    let (owner, probe) = storage(maximum, eos, Behavior::Ready);
    (
        RetainedGenerationSequence::from_retained_storage(owner).unwrap(),
        probe,
    )
}

#[test]
fn retained_and_legacy_share_commit_precedence_and_cancellation() {
    let cases = [
        (vec![7, 7], true, true, FinishReason::StopSequence),
        (vec![7], false, true, FinishReason::GrammarComplete),
        (vec![7], false, false, FinishReason::Eos),
        (vec![], false, false, FinishReason::MaxTokens),
    ];
    for (eos, stop_sequence, grammar_complete, expected) in cases {
        let mut legacy = GenerationSequence::new(1, eos.iter().copied());
        let (retained, probe) = sequence(1, &eos);
        let mut retained = retained.prepare_storage().unwrap();
        let signals = TokenTerminalSignals {
            stop_sequence,
            grammar_complete,
        };
        assert_eq!(retained.commit(7, signals), legacy.commit(7, signals));
        assert_eq!(retained.finish_reason(), Some(expected));
        assert_eq!(retained.tokens(), &[7]);
        assert_eq!(retained.remaining(), 0);
        assert_eq!(retained.cancel(), legacy.cancel());
        assert_eq!(
            retained.commit(8, signals),
            Err(GenerationError::AlreadyFinished)
        );
        let ids = retained.into_token_ids();
        assert_eq!(ids.as_slice(), legacy.into_tokens());
        assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
        drop(ids);
        assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
    }
    let mut legacy = GenerationSequence::new(4, [9]);
    let (retained, _) = sequence(4, &[9]);
    let mut retained = retained.prepare_storage().unwrap();
    for token in [3, 4] {
        assert_eq!(
            retained.commit(token, Default::default()),
            legacy.commit(token, Default::default())
        );
    }
    let cancellation = GenerationCancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        retained.observe_cancellation(&cancellation),
        legacy.observe_cancellation(&cancellation)
    );
    assert_eq!(retained.finish_reason(), Some(FinishReason::Cancelled));
    assert_eq!(retained.into_token_ids().as_slice(), legacy.into_tokens());
}

#[test]
fn cancellation_keeps_zero_and_nonzero_storage_dormant() {
    for maximum in [0, 5] {
        let (mut retained, probe) = sequence(maximum, &[2]);
        let mut legacy = GenerationSequence::new(maximum, [2]);
        assert_eq!(retained.finish_reason(), legacy.finish_reason());
        assert_eq!(retained.cancel(), legacy.cancel());
        assert_eq!(retained.finish_reason(), Some(FinishReason::Cancelled));
        let ids = retained.into_token_ids();
        assert!(ids.is_empty());
        assert_eq!(probe.prepared.load(Ordering::SeqCst), 0);
        assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
        drop(ids);
        assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn retained_preparation_is_required_once_and_freeze_preserves_buffer() {
    let (mut sequence, probe) = sequence(5, &[9]);
    assert_eq!(
        sequence.commit(3, Default::default()),
        Err(GenerationError::StorageNotReady)
    );
    assert!(sequence.tokens().is_empty());
    let mut sequence = sequence
        .prepare_storage()
        .unwrap()
        .prepare_storage()
        .unwrap();
    assert_eq!(probe.prepared.load(Ordering::SeqCst), 1);
    let pointer = sequence.storage.owner.token_slots().as_ptr();
    let capacity = sequence.storage.owner.token_slots().len();
    for token in [3, 9] {
        sequence.commit(token, Default::default()).unwrap();
    }
    assert_eq!(sequence.finish_reason(), Some(FinishReason::Eos));
    let ids = sequence.into_token_ids();
    assert_eq!(ids.as_ptr(), pointer);
    assert_eq!(ids.as_slice(), &[3, 9]);
    assert_eq!(capacity, 5);
    let alias = ids.clone();
    drop(ids);
    assert_eq!(alias.as_ptr(), pointer);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn partial_owning_iterator_retains_custody_and_exact_remaining_length() {
    let (retained, probe) = sequence(4, &[]);
    let mut retained = retained.prepare_storage().unwrap();
    for token in [2, 3, 5, 7] {
        retained.commit(token, Default::default()).unwrap();
    }
    let ids = retained.into_token_ids();
    let alias = ids.clone();
    let mut iter = ids.into_iter();
    drop(alias);
    assert_eq!(iter.len(), 4);
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next_back(), Some(7));
    assert_eq!(iter.size_hint(), (2, Some(2)));
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    drop(iter);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
    let mut empty = GenerationTokenIds::from(vec![]).into_iter();
    assert_eq!(empty.next(), None);
    assert_eq!(empty.next_back(), None);
    assert_eq!(empty.next(), None);
}

#[test]
fn preparation_failure_preserves_original_cause_partial_owner_and_no_retry() {
    let (owner, probe) = storage(8, vec![9], Behavior::Fail);
    let sequence = RetainedGenerationSequence::from_retained_storage(owner).unwrap();
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(
        failure.cause().kind(),
        BackendFailureKind::ResourceExhausted
    );
    assert!(failure.cause().source().unwrap().is::<PreparationFailure>());
    assert!(failure.sequence().tokens().is_empty());
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    let mut sequence = failure.into_sequence();
    assert_eq!(probe.error_retired.load(Ordering::SeqCst), 1);
    assert_eq!(probe.error_without_owner.load(Ordering::SeqCst), 0);
    assert_eq!(
        sequence.commit(1, Default::default()),
        Err(GenerationError::StorageNotReady)
    );
    let failure = sequence.prepare_storage().unwrap_err();
    assert_eq!(probe.prepared.load(Ordering::SeqCst), 1);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    drop(failure);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn bad_policy_and_wrong_prepared_extent_preserve_owner() {
    let (owner, probe) = storage(3, vec![8, 2, 2], Behavior::Ready);
    let pointer = Arc::as_ptr(&owner.payload);
    let failure = RetainedGenerationSequence::from_retained_storage(owner).unwrap_err();
    let owner = failure.into_storage();
    assert_eq!(owner.eos_token_ids(), &[8, 2, 2]);
    assert_eq!(probe.prepared.load(Ordering::SeqCst), 0);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    // The returned trait object still retains the same concrete payload.
    let ids = owner.into_token_ids(0);
    assert_eq!(
        Arc::as_ptr(ids.owner.as_ref().unwrap()).cast::<()>(),
        pointer.cast::<()>()
    );
    drop(ids);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);

    let (owner, probe) = storage(3, vec![], Behavior::WrongLength);
    let sequence = RetainedGenerationSequence::from_retained_storage(owner).unwrap();
    let failure = sequence.prepare_storage().unwrap_err();
    assert!(failure.cause().source().unwrap().is::<GenerationError>());
    assert!(failure.sequence().tokens().is_empty());
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    drop(failure);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn preparation_unwind_preserves_exact_payload_and_retires_partial_owner() {
    let (owner, probe) = storage(5, vec![], Behavior::Panic);
    let sequence = RetainedGenerationSequence::from_retained_storage(owner).unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = sequence.prepare_storage();
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"exact slot preparation panic")
    );
    assert_eq!(probe.prepared.load(Ordering::SeqCst), 1);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn legacy_clone_extraction_and_public_send_sync_remain_available() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<GenerationSequence>();
    send_sync::<RetainedGenerationSequence>();
    send_sync::<RetainedGenerationStorageOwner>();
    send_sync::<GenerationTokenIds>();
    send_sync::<GenerationTokenIdsIntoIter>();
    send_sync::<RetainedSequencePreparationError>();
    let mut sequence = GenerationSequence::new(5, [9, 9, 7]);
    sequence.commit(2, Default::default()).unwrap();
    let pointer = sequence.tokens().as_ptr();
    let mut alias = sequence.clone();
    alias.commit(3, Default::default()).unwrap();
    assert_eq!(sequence.tokens(), &[2]);
    assert_eq!(alias.into_tokens(), vec![2, 3]);
    let tokens = sequence.into_tokens();
    assert_eq!(tokens.as_ptr(), pointer);
    let immutable = GenerationTokenIds::from(tokens);
    assert_eq!(immutable.as_ptr(), pointer);
    assert_eq!(immutable, GenerationTokenIds::from(vec![2]));
    assert_eq!(
        (&immutable).into_iter().copied().collect::<Vec<_>>(),
        vec![2]
    );
}

#[test]
fn shortened_sequence_keeps_physical_capacity_and_terminal_precedence() {
    for committed in [0, 1] {
        let (mut retained, probe) = sequence(4, &[99]);
        let mut legacy = GenerationSequence::new(4, [99]);
        if committed == 1 {
            retained = retained.prepare_storage().unwrap();
            assert_eq!(
                retained.commit(13, Default::default()),
                legacy.commit(13, Default::default())
            );
        }
        retained.restrict_remaining(2);
        legacy.restrict_remaining(2);
        retained.restrict_remaining(usize::MAX);
        legacy.restrict_remaining(usize::MAX);
        assert_eq!(retained.max_tokens(), committed + 2);
        retained = retained.prepare_storage().unwrap();
        assert_eq!(retained.storage.owner.max_tokens(), 4);
        assert_eq!(retained.storage.owner.token_slots().len(), 4);
        for token in [17, 23] {
            assert_eq!(
                retained.commit(token, Default::default()),
                legacy.commit(token, Default::default())
            );
        }
        assert_eq!(retained.finish_reason(), Some(FinishReason::MaxTokens));
        retained.restrict_remaining(usize::MAX);
        assert_eq!(
            retained.commit(42, Default::default()),
            Err(GenerationError::AlreadyFinished)
        );
        let ids = retained.into_token_ids();
        assert_eq!(ids.as_slice(), legacy.into_tokens());
        assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
        drop(ids);
        assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
    }
    let (mut retained, _) = sequence(4, &[99]);
    retained.restrict_remaining(1);
    let mut retained = retained.prepare_storage().unwrap();
    assert_eq!(
        retained
            .commit(99, Default::default())
            .unwrap()
            .finish_reason,
        Some(FinishReason::Eos)
    );
    retained.restrict_remaining(0);
    assert_eq!(retained.finish_reason(), Some(FinishReason::Eos));
    let (owner, probe) = storage(4, vec![], Behavior::WrongLength);
    let mut malformed = RetainedGenerationSequence::from_retained_storage(owner).unwrap();
    malformed.restrict_remaining(2);
    let failure = malformed.prepare_storage().unwrap_err();
    assert_eq!(probe.retired.load(Ordering::SeqCst), 0);
    drop(failure);
    assert_eq!(probe.retired.load(Ordering::SeqCst), 1);
}

#[test]
fn shortened_dormant_sequence_copy_preserves_paid_capacity() {
    let (mut source, source_probe) = sequence(4, &[]);
    source.restrict_remaining(2);
    let (destination, destination_probe) = storage(4, vec![], Behavior::Ready);
    let copied = source
        .prepare_host_copy()
        .complete(RetainedGenerationStorageOwner::new(destination))
        .unwrap();
    assert_eq!(copied.max_tokens(), 2);
    let mut copied = copied.prepare_storage().unwrap();
    assert_eq!(copied.storage.owner.token_slots().len(), 4);
    assert_eq!(
        copied.commit(17, Default::default()).unwrap().finish_reason,
        None
    );
    assert_eq!(
        copied.commit(23, Default::default()).unwrap().finish_reason,
        Some(FinishReason::MaxTokens)
    );
    assert!(source.tokens().is_empty());
    drop(copied);
    assert_eq!(destination_probe.retired.load(Ordering::SeqCst), 1);
    assert_eq!(source_probe.retired.load(Ordering::SeqCst), 0);
    source.restrict_remaining(0);
    assert_eq!(source.finish_reason(), Some(FinishReason::MaxTokens));
    assert_eq!(source_probe.prepared.load(Ordering::SeqCst), 0);
}
