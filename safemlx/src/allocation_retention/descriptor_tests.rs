//! A real native deferred-owner callback pauses known-input descriptor
//! retirement while another host invokes the actual global Rust reclaimer.
//! This is lifetime evidence, not a byte bound or managed admission fixture.
use super::*;
use crate::{Device, DeviceType, Stream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    armed: AtomicBool,
    child_entered: AtomicBool,
    release_child: AtomicBool,
    child_callback_returned: AtomicBool,
    reclaim_requested: AtomicBool,
    first_reclaim_finished: AtomicBool,
    finish_reclaim: AtomicBool,
    owner_dropped: AtomicBool,
    owner_saw_completed_child_callback: AtomicBool,
}

unsafe extern "C" fn pause_child_retirement(payload: *mut c_void) {
    // SAFETY: successful native deferred attachment owns exactly this Box and
    // invokes the callback once when its final descriptor/backing owner retires.
    let state = unsafe { Box::from_raw(payload.cast::<Arc<State>>()) };
    if state.armed.load(Ordering::SeqCst) {
        state.child_entered.store(true, Ordering::SeqCst);
        while !state.release_child.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    }
    state.child_callback_returned.store(true, Ordering::SeqCst);
}

struct ReleaseWaiters(Arc<State>);
impl Drop for ReleaseWaiters {
    fn drop(&mut self) {
        self.0.release_child.store(true, Ordering::SeqCst);
        self.0.reclaim_requested.store(true, Ordering::SeqCst);
        self.0.finish_reclaim.store(true, Ordering::SeqCst);
    }
}

struct DeferredAuthority(Arc<State>);
impl Drop for DeferredAuthority {
    fn drop(&mut self) {
        self.0.owner_saw_completed_child_callback.store(
            self.0.child_callback_returned.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
        self.0.owner_dropped.store(true, Ordering::SeqCst);
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !predicate() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::yield_now();
    }
    true
}

#[test]
fn descriptor_parent_authority_outlives_blocked_input_and_concurrent_rust_reclaimer() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let state = Arc::new(State::default());
    let input = Array::from_slice(&[2_f32, 3.0], &[2])
        .square(&stream)
        .unwrap();
    assert_eq!(input.allocation_info().unwrap(), None);
    let payload = Box::into_raw(Box::new(Arc::clone(&state)));
    let mut attached = false;
    let status = {
        let _guard = runtime_lock::enter();
        // SAFETY: the callback owns no native/source roots, the input remains
        // live, and attached is the existing sole ownership-transfer result.
        unsafe {
            safemlx_sys::mlx_array_retain_deferred_allocation_owner(
                &mut attached,
                input.as_ptr(),
                payload.cast(),
                Some(pause_child_retirement),
            )
        }
    };
    if !attached {
        // SAFETY: rejection leaves the original payload with this caller.
        drop(unsafe { Box::from_raw(payload) });
    }
    assert_eq!(status, 0);
    assert!(attached);
    let root = input.exp(&stream).unwrap();
    root.retain_deferred_allocation_owner(DeferredAuthority(Arc::clone(&state)))
        .unwrap();
    let alias = root.clone();
    drop((input, root));
    reclaim_allocation_owners();
    assert!(!state.owner_dropped.load(Ordering::SeqCst));
    assert_eq!(alias.allocation_info().unwrap(), None);

    // Declared after the last local native owner, so unwind opens the gates
    // before that owner can invoke its callback (including thread-spawn error).
    let _release_waiters = ReleaseWaiters(Arc::clone(&state));
    state.armed.store(true, Ordering::SeqCst);
    let reclaim_state = Arc::clone(&state);
    let reclaimer = std::thread::spawn(move || {
        while !reclaim_state.reclaim_requested.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        reclaim_allocation_owners();
        reclaim_state
            .first_reclaim_finished
            .store(true, Ordering::SeqCst);
        while !reclaim_state.finish_reclaim.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        wait_until(|| {
            reclaim_allocation_owners();
            reclaim_state.owner_dropped.load(Ordering::SeqCst)
        })
    });
    let dropper = std::thread::spawn(move || drop(alias));
    let entered = wait_until(|| state.child_entered.load(Ordering::SeqCst));
    state.reclaim_requested.store(true, Ordering::SeqCst);
    let observed = wait_until(|| state.first_reclaim_finished.load(Ordering::SeqCst));
    let premature = state.owner_dropped.load(Ordering::SeqCst);
    // Always release every waiter and join before assertions, including timeout
    // and worker panic paths. A test failure must not strand native retirement.
    state.release_child.store(true, Ordering::SeqCst);
    let drop_result = dropper.join();
    state.finish_reclaim.store(true, Ordering::SeqCst);
    let reclaim_result = reclaimer.join();
    assert!(entered && observed);
    assert!(
        !premature,
        "authority was globally reclaimable while its known input still retired"
    );
    drop_result.unwrap();
    assert!(reclaim_result.unwrap());
    assert!(state.child_callback_returned.load(Ordering::SeqCst));
    assert!(state
        .owner_saw_completed_child_callback
        .load(Ordering::SeqCst));
}
