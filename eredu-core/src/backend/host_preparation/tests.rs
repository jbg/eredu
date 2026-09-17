use super::*;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex, OnceLock, Weak,
    },
};

// Deliberately neither Clone nor Debug: authority aliases share one token and
// do not need its implementation to support either operation.
struct DropCount(Arc<AtomicUsize>);

impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn authority_clones_share_one_token_until_final_retirement_on_another_thread() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HostPreparationAuthority>();

    let drops = Arc::new(AtomicUsize::new(0));
    let original = HostPreparationAuthority::retain(DropCount(Arc::clone(&drops)));
    let first = original.clone();
    let last = first.clone();
    let _ = format!("{original:?}");
    drop(original);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    std::thread::spawn(move || drop(last)).join().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct Payload {
    storage: Arc<Vec<u8>>,
    active: Arc<AtomicBool>,
    drops: Arc<AtomicUsize>,
}

impl Drop for Payload {
    fn drop(&mut self) {
        assert!(self.active.load(Ordering::SeqCst));
        assert_eq!(self.storage.as_slice(), &[7, 11, 13]);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct Exclusion {
    active: Arc<AtomicBool>,
    payload: Arc<OnceLock<Weak<Vec<u8>>>>,
    payload_drops: Arc<AtomicUsize>,
    token_drops: Arc<AtomicUsize>,
}

impl Drop for Exclusion {
    fn drop(&mut self) {
        assert!(
            self.payload.get().unwrap().upgrade().is_none(),
            "payload must retire first"
        );
        assert_eq!(self.payload_drops.load(Ordering::SeqCst), 1);
        assert!(self.active.swap(false, Ordering::SeqCst));
        self.token_drops.fetch_add(1, Ordering::SeqCst);
    }
}

// This is an explicit caller composition, not a public arbitrary-payload
// wrapper or a completeness assertion made by HostPreparationAuthority.
struct OwnedPayload {
    _payload: Payload,
    _authority: HostPreparationAuthority,
}

fn owned_payload(
    active: &Arc<AtomicBool>,
    payload_drops: &Arc<AtomicUsize>,
    token_drops: &Arc<AtomicUsize>,
) -> OwnedPayload {
    let payload = Arc::new(OnceLock::new());
    let authority = HostPreparationAuthority::retain(Exclusion {
        active: Arc::clone(active),
        payload: Arc::clone(&payload),
        payload_drops: Arc::clone(payload_drops),
        token_drops: Arc::clone(token_drops),
    });
    let storage = Arc::new(vec![7, 11, 13]);
    payload.set(Arc::downgrade(&storage)).unwrap();
    OwnedPayload {
        _payload: Payload {
            storage,
            active: Arc::clone(active),
            drops: Arc::clone(payload_drops),
        },
        _authority: authority,
    }
}

#[test]
fn explicit_payload_then_authority_composition_preserves_exclusion_during_drop_and_unwind() {
    for unwind in [false, true] {
        let active = Arc::new(AtomicBool::new(true));
        let payload_drops = Arc::new(AtomicUsize::new(0));
        let token_drops = Arc::new(AtomicUsize::new(0));
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let payload = owned_payload(&active, &payload_drops, &token_drops);
            if unwind {
                panic!("host constructor failed after creating a covered payload");
            }
            drop(payload);
        }));
        assert_eq!(outcome.is_err(), unwind);
        assert!(!active.load(Ordering::SeqCst));
        assert_eq!(payload_drops.load(Ordering::SeqCst), 1);
        assert_eq!(token_drops.load(Ordering::SeqCst), 1);
    }
}

struct ReentrantDrop {
    nested: Arc<Mutex<Option<HostPreparationAuthority>>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for ReentrantDrop {
    fn drop(&mut self) {
        let nested = self.nested.try_lock().unwrap().take();
        drop(nested);
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn final_token_drop_can_reenter_external_custody_and_retire_another_authority() {
    let first_drops = Arc::new(AtomicUsize::new(0));
    let second_drops = Arc::new(AtomicUsize::new(0));
    let nested = Arc::new(Mutex::new(Some(HostPreparationAuthority::retain(
        DropCount(Arc::clone(&second_drops)),
    ))));
    let authority = HostPreparationAuthority::retain(ReentrantDrop {
        nested: Arc::clone(&nested),
        drops: Arc::clone(&first_drops),
    });
    let alias = authority.clone();
    drop(authority);
    assert_eq!(first_drops.load(Ordering::SeqCst), 0);
    assert_eq!(second_drops.load(Ordering::SeqCst), 0);
    drop(alias);
    assert_eq!(first_drops.load(Ordering::SeqCst), 1);
    assert_eq!(second_drops.load(Ordering::SeqCst), 1);
    assert!(nested.lock().unwrap().is_none());
}
