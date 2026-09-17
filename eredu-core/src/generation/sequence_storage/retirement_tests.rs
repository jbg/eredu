use super::*;
use crate::BackendFailureKind;
use std::{
    error::Error as _,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst},
        Barrier,
    },
};

// These probes establish dispatch/lifetime order, not allocator observations or
// funding. The concrete Box and Arc deallocation argument uses their documented
// owned operations, with every strong exit closed below and no Weak exposed.
#[derive(Debug, Default)]
struct Probe {
    box_retire: AtomicUsize,
    freeze: AtomicUsize,
    arc_retire: AtomicUsize,
    payload_retire: AtomicUsize,
    slots_drop: AtomicUsize,
    custody_drop: AtomicUsize,
    cause_drop: AtomicUsize,
    wrong_order: AtomicUsize,
    prepared: AtomicUsize,
    panic_read: AtomicBool,
    panic_cause: AtomicBool,
}
#[derive(Debug)]
struct Slots {
    values: Vec<u32>,
    probe: Arc<Probe>,
}
impl Drop for Slots {
    fn drop(&mut self) {
        self.probe.slots_drop.fetch_add(1, SeqCst);
    }
}
#[derive(Debug)]
struct Custody(Arc<Probe>);
impl Drop for Custody {
    fn drop(&mut self) {
        if self.0.slots_drop.load(SeqCst) != 1 {
            self.0.wrong_order.fetch_add(1, SeqCst);
        }
        self.0.custody_drop.fetch_add(1, SeqCst);
    }
}
#[derive(Debug)]
struct Payload {
    slots: Slots,
    eos: Vec<u32>,
    committed: usize,
    custody: Custody,
}
impl GenerationTokenIdStorage for Payload {
    fn token_ids(&self) -> &[u32] {
        if self.custody.0.panic_read.load(SeqCst) {
            std::panic::panic_any("token callback panic");
        }
        &self.slots.values[..self.committed]
    }
    fn retire(self: Arc<Self>) {
        self.custody.0.arc_retire.fetch_add(1, SeqCst);
        if let Some(payload) = Arc::into_inner(self) {
            // No Weak exists. The concrete Arc allocation is already gone;
            // payload still owns all slots/EOS and its final custody.
            payload.custody.0.payload_retire.fetch_add(1, SeqCst);
            drop(payload);
        }
    }
}

// The dormant/mutable provider must close its Arc exits too, including callback
// unwind. It never exports a Weak or an ordinary-dropping strong alias.
#[derive(Debug)]
struct PayloadOwner(Option<Arc<Payload>>);
impl PayloadOwner {
    fn get(&self) -> &Payload {
        self.0.as_deref().unwrap()
    }
    fn get_mut(&mut self) -> &mut Payload {
        Arc::get_mut(self.0.as_mut().unwrap()).expect("sole mutable payload")
    }
    fn freeze(mut self, committed: usize) -> GenerationTokenIds {
        self.get_mut().committed = committed;
        GenerationTokenIds::from_owner(self.0.take().unwrap())
    }
}
impl Drop for PayloadOwner {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Behavior {
    Ready,
    EosPanic,
    MaximumPanic,
    DebugPanic,
    PreparationPanic,
    PreparationFailure,
    WrongLength,
    FreezePanic,
}
struct Provider {
    maximum: usize,
    behavior: Behavior,
    probe: Arc<Probe>,
    payload: PayloadOwner,
}
impl fmt::Debug for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if matches!(self.behavior, Behavior::DebugPanic) {
            std::panic::panic_any("debug callback panic");
        }
        f.debug_struct("Provider")
            .field("maximum", &self.maximum)
            .finish_non_exhaustive()
    }
}
// The function returns the moved concrete value only after the Box allocation
// has been deallocated. Fallible work and payload destruction happen afterward.
fn unbox<T>(owner: Box<T>) -> T {
    *owner
}
#[derive(Debug)]
struct PartialFailure(Arc<Probe>);
impl fmt::Display for PartialFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("owned partial preparation")
    }
}
impl std::error::Error for PartialFailure {}
impl Drop for PartialFailure {
    fn drop(&mut self) {
        if self.0.custody_drop.load(SeqCst) != 0 {
            self.0.wrong_order.fetch_add(1, SeqCst);
        }
        self.0.cause_drop.fetch_add(1, SeqCst);
        if self.0.panic_cause.load(SeqCst) {
            std::panic::panic_any("cause disposal panic");
        }
    }
}
impl RetainedGenerationStorage for Provider {
    fn max_tokens(&self) -> usize {
        if matches!(self.behavior, Behavior::MaximumPanic) {
            std::panic::panic_any("maximum callback panic");
        }
        self.maximum
    }
    fn eos_token_ids(&self) -> &[u32] {
        if matches!(self.behavior, Behavior::EosPanic) {
            std::panic::panic_any("eos callback panic");
        }
        &self.payload.get().eos
    }
    fn prepare_tokens(&mut self) -> Result<(), BackendFailure> {
        self.probe.prepared.fetch_add(1, SeqCst);
        let extent = match self.behavior {
            Behavior::PreparationFailure | Behavior::PreparationPanic => 1,
            Behavior::WrongLength => self.maximum + 1,
            _ => self.maximum,
        };
        self.payload.get_mut().slots.values.resize(extent, 0);
        match self.behavior {
            Behavior::PreparationPanic => std::panic::panic_any("preparation callback panic"),
            Behavior::PreparationFailure => Err(BackendFailure::new(
                BackendFailureKind::ResourceExhausted,
                PartialFailure(Arc::clone(&self.probe)),
            )),
            _ => Ok(()),
        }
    }
    fn token_slots(&self) -> &[u32] {
        &self.payload.get().slots.values
    }
    fn token_slots_mut(&mut self) -> &mut [u32] {
        &mut self.payload.get_mut().slots.values
    }
    fn into_token_ids(self: Box<Self>, committed: usize) -> GenerationTokenIds {
        let provider = unbox(self);
        provider.probe.freeze.fetch_add(1, SeqCst);
        if matches!(provider.behavior, Behavior::FreezePanic) {
            std::panic::panic_any("freeze callback panic");
        }
        provider.payload.freeze(committed)
    }
    fn retire(self: Box<Self>) {
        let provider = unbox(self);
        provider.probe.box_retire.fetch_add(1, SeqCst);
        drop(provider);
    }
}
fn provider(maximum: usize, eos: &[u32], behavior: Behavior) -> (Box<Provider>, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let owner = Box::new(Provider {
        maximum,
        behavior,
        probe: Arc::clone(&probe),
        payload: PayloadOwner(Some(Arc::new(Payload {
            slots: Slots {
                values: Vec::new(),
                probe: Arc::clone(&probe),
            },
            eos: eos.to_vec(),
            committed: 0,
            custody: Custody(Arc::clone(&probe)),
        }))),
    });
    (owner, probe)
}
fn dormant(behavior: Behavior) -> (RetainedGenerationSequence, Arc<Probe>) {
    let (owner, probe) = provider(4, &[9], behavior);
    (
        RetainedGenerationSequence::from_retained_storage(owner).unwrap(),
        probe,
    )
}
fn retired(probe: &Probe, boxes: usize, freezes: usize, arcs: usize) {
    assert_eq!(probe.box_retire.load(SeqCst), boxes);
    assert_eq!(probe.freeze.load(SeqCst), freezes);
    assert_eq!(probe.arc_retire.load(SeqCst), arcs);
    assert_eq!(probe.payload_retire.load(SeqCst), 1);
    assert_eq!(probe.slots_drop.load(SeqCst), 1);
    assert_eq!(probe.custody_drop.load(SeqCst), 1);
    assert_eq!(probe.wrong_order.load(SeqCst), 0);
}
fn panic_is(action: impl FnOnce(), expected: &'static str) {
    let cause = catch_unwind(AssertUnwindSafe(action)).unwrap_err();
    assert_eq!(cause.downcast_ref::<&'static str>(), Some(&expected));
}

#[test]
fn owned_retirement_wraps_before_constructor_and_debug_callbacks() {
    for (behavior, cause) in [
        (Behavior::EosPanic, "eos callback panic"),
        (Behavior::MaximumPanic, "maximum callback panic"),
    ] {
        let (owner, probe) = provider(4, &[9], behavior);
        panic_is(
            || {
                let _ = RetainedGenerationSequence::from_retained_storage(owner);
            },
            cause,
        );
        retired(&probe, 1, 0, 1);
    }
    let (sequence, probe) = dormant(Behavior::DebugPanic);
    panic_is(
        || {
            let sequence = sequence;
            let _ = format!("{sequence:?}");
        },
        "debug callback panic",
    );
    retired(&probe, 1, 0, 1);
}

#[test]
fn owned_retirement_construction_error_returns_closed_unchanged_owner() {
    let (owner, probe) = provider(4, &[9, 2, 2], Behavior::Ready);
    let error = RetainedGenerationSequence::from_retained_storage(owner).unwrap_err();
    let owner = error.into_storage();
    assert_eq!(owner.max_tokens(), 4);
    assert_eq!(owner.eos_token_ids(), &[9, 2, 2]);
    assert_eq!(probe.box_retire.load(SeqCst), 0);
    let error = RetainedGenerationSequence::from_retained_owner(owner).unwrap_err();
    assert_eq!(probe.custody_drop.load(SeqCst), 0);
    drop(error);
    retired(&probe, 1, 0, 1);
}

#[test]
fn owned_retirement_handles_dormant_ready_cancelled_and_zero_owners() {
    for prepared in [false, true] {
        let (mut sequence, probe) = dormant(Behavior::Ready);
        if prepared {
            sequence = sequence.prepare_storage().unwrap();
            sequence.commit(7, Default::default()).unwrap();
        }
        sequence.cancel();
        drop(sequence);
        retired(&probe, 1, 0, 1);
        assert_eq!(probe.prepared.load(SeqCst), usize::from(prepared));
    }
    for maximum in [0, 4] {
        let (owner, probe) = provider(maximum, &[], Behavior::Ready);
        let mut sequence = RetainedGenerationSequence::from_retained_storage(owner).unwrap();
        sequence.cancel();
        let tokens = sequence.into_token_ids();
        assert!(tokens.is_empty());
        assert_eq!(probe.prepared.load(SeqCst), 0);
        assert_eq!(probe.custody_drop.load(SeqCst), 0);
        drop(tokens);
        retired(&probe, 0, 1, 1);
    }
}

#[test]
fn owned_retirement_preserves_partial_error_and_attempt_before_disposal() {
    for recover in [false, true] {
        let (sequence, probe) = dormant(Behavior::PreparationFailure);
        let error = sequence.prepare_storage().unwrap_err();
        assert!(error.cause().source().unwrap().is::<PartialFailure>());
        assert_eq!(probe.custody_drop.load(SeqCst), 0);
        if recover {
            let sequence = error.into_sequence();
            assert_eq!(probe.cause_drop.load(SeqCst), 1);
            let error = sequence.prepare_storage().unwrap_err();
            assert!(error.cause().source().unwrap().is::<GenerationError>());
            assert_eq!(probe.prepared.load(SeqCst), 1);
            drop(error);
        } else {
            drop(error);
        }
        retired(&probe, 1, 0, 1);
        assert_eq!(probe.cause_drop.load(SeqCst), 1);
    }
    let (sequence, probe) = dormant(Behavior::WrongLength);
    let error = sequence.prepare_storage().unwrap_err();
    assert!(error.cause().source().unwrap().is::<GenerationError>());
    drop(error);
    retired(&probe, 1, 0, 1);
}

#[test]
fn owned_retirement_closes_preparation_and_consuming_freeze_panics() {
    let (sequence, probe) = dormant(Behavior::PreparationPanic);
    panic_is(
        || {
            let _ = sequence.prepare_storage();
        },
        "preparation callback panic",
    );
    retired(&probe, 1, 0, 1);
    let (sequence, probe) = dormant(Behavior::FreezePanic);
    let mut sequence = sequence.prepare_storage().unwrap();
    sequence.commit(7, Default::default()).unwrap();
    panic_is(
        || {
            let _ = sequence.into_token_ids();
        },
        "freeze callback panic",
    );
    retired(&probe, 0, 1, 1);
}

#[test]
fn owned_retirement_freeze_and_partial_iterator_keep_original_buffers() {
    let (sequence, probe) = dormant(Behavior::Ready);
    let mut sequence = sequence.prepare_storage().unwrap();
    for token in [2, 3, 5, 7] {
        sequence.commit(token, Default::default()).unwrap();
    }
    let pointer = sequence.tokens().as_ptr();
    let tokens = sequence.into_token_ids();
    assert_eq!(tokens.as_ptr(), pointer);
    assert_eq!(tokens.as_slice(), &[2, 3, 5, 7]);
    let alias = tokens.clone();
    let mut iter = tokens.into_iter();
    drop(alias);
    assert_eq!(probe.arc_retire.load(SeqCst), 1);
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next_back(), Some(7));
    assert_eq!(iter.len(), 2);
    assert_eq!(probe.custody_drop.load(SeqCst), 0);
    drop(iter);
    retired(&probe, 0, 1, 2);
}

#[test]
fn owned_retirement_concurrent_final_aliases_extract_one_actual_payload() {
    for _ in 0..16 {
        let (sequence, probe) = dormant(Behavior::Ready);
        let mut sequence = sequence.prepare_storage().unwrap();
        sequence.commit(7, Default::default()).unwrap();
        let tokens = sequence.into_token_ids();
        let barrier = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            for _ in 0..7 {
                let tokens = tokens.clone();
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    assert_eq!(tokens.as_slice(), &[7]);
                    barrier.wait();
                    drop(tokens);
                });
            }
            barrier.wait();
            drop(tokens);
        });
        retired(&probe, 0, 1, 8);
    }
}

#[test]
fn owned_retirement_keeps_dispatch_during_token_read_and_iterator_unwind() {
    for iterator in [false, true] {
        let (sequence, probe) = dormant(Behavior::Ready);
        let mut sequence = sequence.prepare_storage().unwrap();
        sequence.commit(7, Default::default()).unwrap();
        let tokens = sequence.into_token_ids();
        let alias = tokens.clone();
        if iterator {
            let mut iter = tokens.into_iter();
            probe.panic_read.store(true, SeqCst);
            panic_is(
                || {
                    let _ = iter.next();
                    drop(iter);
                },
                "token callback panic",
            );
        } else {
            probe.panic_read.store(true, SeqCst);
            panic_is(
                || {
                    let _ = tokens.as_slice();
                    drop(tokens);
                },
                "token callback panic",
            );
        }
        assert_eq!(probe.arc_retire.load(SeqCst), 1);
        assert_eq!(probe.custody_drop.load(SeqCst), 0);
        probe.panic_read.store(false, SeqCst);
        assert_eq!(alias.as_slice(), &[7]);
        drop(alias);
        retired(&probe, 0, 1, 2);
    }
}

#[test]
fn owned_retirement_preserves_storage_when_partial_cause_disposal_panics() {
    for recover in [false, true] {
        let (sequence, probe) = dormant(Behavior::PreparationFailure);
        let error = sequence.prepare_storage().unwrap_err();
        probe.panic_cause.store(true, SeqCst);
        panic_is(
            || {
                if recover {
                    let _ = error.into_sequence();
                } else {
                    drop(error);
                }
            },
            "cause disposal panic",
        );
        assert_eq!(probe.cause_drop.load(SeqCst), 1);
        retired(&probe, 1, 0, 1);
    }
}
