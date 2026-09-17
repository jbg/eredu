//! Exact allocation witness, compiled only into safemlx's unit-test binary.
use super::*;
use std::{
    alloc::{GlobalAlloc, System},
    sync::{atomic::AtomicBool, Mutex, MutexGuard},
};

#[derive(Debug)]
struct WitnessAllocator;
#[global_allocator]
static ALLOCATOR: WitnessAllocator = WitnessAllocator;
static WATCHED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicBool = AtomicBool::new(false);
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: every operation delegates the exact pointer/layout arguments to
// System. Tracking uses only atomics and constant thread-local Cells, never
// allocation, callbacks or locks.
unsafe impl GlobalAlloc for WitnessAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        crate::utils::allocation_test::allocated();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        crate::utils::allocation_test::allocated();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        crate::utils::allocation_test::allocated();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let watched = ptr.addr() == WATCHED.load(Ordering::Acquire);
        unsafe { System.dealloc(ptr, layout) };
        // Set only after the actual allocator returned. The watched live Box
        // cannot share its address with any unrelated concurrent allocation.
        if watched {
            FREED.store(true, Ordering::Release);
        }
    }
}

struct Witness {
    _serial: MutexGuard<'static, ()>,
}
impl Witness {
    fn new() -> Self {
        let serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
        WATCHED.store(0, Ordering::Release);
        FREED.store(false, Ordering::Release);
        Self { _serial: serial }
    }
    fn watch<T: Send + 'static>(&self, prepared: &PreparedAllocationOwner<T>) {
        let ptr = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()).addr();
        WATCHED.store(ptr, Ordering::Release);
    }
}
impl Drop for Witness {
    fn drop(&mut self) {
        WATCHED.store(0, Ordering::Release);
        FREED.store(false, Ordering::Release);
    }
}
struct NodeProbe(Arc<AtomicUsize>);
impl Drop for NodeProbe {
    fn drop(&mut self) {
        assert!(
            FREED.load(Ordering::Acquire),
            "Rust node outlived its accounting owner"
        );
        if !std::thread::panicking() {
            assert!(runtime_lock::can_reclaim_submission_resources());
        }
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn actual_rust_node_retires_before_pending_error_unwind_and_attached_owner() {
    for case in 0..4 {
        // Declared before the prepared value so tracking resets only after its
        // cleanup, including unwind from any assertion below.
        let witness = Witness::new();
        let drops = Arc::new(AtomicUsize::new(0));
        let prepared = PreparedAllocationOwner::try_new(NodeProbe(drops.clone())).unwrap();
        witness.watch(&prepared);
        match case {
            0 => drop(prepared),
            1 => {
                let input = Array::from_int(3);
                let error = prepared
                    .attach_with(|out, node, payload| unsafe {
                        safemlx_sys::mlx_array_attach_prepared_allocation_owner(
                            out,
                            input.as_ptr(),
                            node,
                            payload,
                            None,
                        )
                    })
                    .unwrap_err();
                assert!(matches!(
                    error.cause(),
                    PreparedAllocationOwnerCause::Native(_)
                ));
                drop(error);
            }
            2 => {
                let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let _prepared = prepared;
                    std::panic::panic_any(91_usize);
                }))
                .unwrap_err();
                assert_eq!(*panic.downcast::<usize>().unwrap(), 91);
            }
            3 => {
                let input = Array::from_int(5);
                prepared.try_attach(&input).unwrap();
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                drop(input);
                wait(&drops, 1, &stream());
            }
            _ => unreachable!(),
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(FREED.load(Ordering::Acquire));
        drop(witness);
        assert_eq!(WATCHED.load(Ordering::Acquire), 0);
    }
}
