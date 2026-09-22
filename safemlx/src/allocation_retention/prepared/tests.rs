use super::*;
use crate::{
    ops::indexing::TryIndexOp, reclaim_allocation_owners, Device, DeviceType, Dtype,
    HostTransferBuffer, HostTransferPolicy, Stream,
};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(runtime_lock::can_reclaim_submission_resources());
            // A real reentrant native query is safe only after the outer lock ends.
            let a = Array::from_slice(&[13_i32], &[1]);
            assert_eq!(a.nbytes(), 4);
        }
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn wait(drops: &Arc<AtomicUsize>, expected: usize, stream: &Stream) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        stream.synchronize().unwrap();
        crate::memory::clear_cache().unwrap();
        reclaim_allocation_owners();
        if drops.load(Ordering::SeqCst) == expected {
            break;
        }
        assert!(Instant::now() < deadline, "prepared owner did not retire");
        std::thread::yield_now();
    }
}
thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|x| x.set(x.get() + 1));
}
struct Unhook;
impl Drop for Unhook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(hook);
    }
}

#[test]
fn prepared_layout_and_construction_need_no_native_entry_or_housekeeping() {
    let retired = Arc::new(AtomicUsize::new(0));
    let queued = Array::from_slice(&[8_i32], &[1]);
    queued.evaluated().unwrap();
    queued
        .retain_allocation_owner(Probe(retired.clone()))
        .unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        drop(queued); // queues its owner while reclamation is forbidden
        ready_tx.send(()).unwrap();
        done_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    runtime_lock::register_housekeeping_hook(hook);
    let _hook = Unhook;
    HOOKS.with(|x| x.set(0));
    let facts = PreparedAllocationOwner::<[u64; 17]>::layout();
    let prepared = PreparedAllocationOwner::try_new([7_u64; 17]);
    done_tx.send(()).unwrap();
    thread.join().unwrap();
    let prepared = prepared.unwrap();
    assert_eq!(prepared.into_owner(), [7; 17]);
    assert_eq!(
        facts.rust_node_bytes(),
        mem::size_of::<OwnedNode<[u64; 17]>>()
    );
    assert_eq!(
        facts.prepared_bytes(),
        mem::size_of::<PreparedAllocationOwner<[u64; 17]>>()
    );
    assert_eq!(
        facts.preparation_failure_bytes(),
        mem::size_of::<PreparedAllocationOwnerError<[u64; 17]>>()
    );
    assert_eq!(
        facts.attachment_failure_bytes(),
        mem::size_of::<PreparedAllocationOwnerError<PreparedAllocationOwner<[u64; 17]>>>()
    );
    assert_eq!(
        facts.allocation_bytes(),
        facts
            .rust_node_bytes()
            .checked_add(facts.native_node_bytes())
    );
    assert_eq!(
        facts.preparation_control_bytes(),
        mem::size_of::<Preparation<[u64; 17]>>()
            + mem::size_of::<Layout>()
            + mem::size_of::<Option<NonNull<u8>>>()
    );
    assert_eq!(facts.native_list_bytes(), 2 * mem::size_of::<usize>());
    assert!(facts.native_node_bytes() > facts.native_list_bytes());
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(
        retired.load(Ordering::SeqCst),
        0,
        "cold preparation must not reap queued owners"
    );
    drop(_hook);
    crate::memory::clear_cache().unwrap();
    reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn failed_node_allocations_preserve_the_exact_original_owner() {
    for fail_native in [true, false] {
        let drops = Arc::new(AtomicUsize::new(0));
        let error = PreparedAllocationOwner::prepare_with(
            Probe(drops.clone()),
            || {
                if fail_native {
                    std::ptr::null_mut()
                } else {
                    unsafe { safemlx_sys::mlx_allocation_owner_node_new() }
                }
            },
            |_| std::ptr::null_mut(),
        )
        .unwrap_err();
        assert!(matches!(
            error.cause(),
            PreparedAllocationOwnerCause::AllocationFailed
        ));
        assert!(Arc::ptr_eq(&error.owner().0, &drops));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn busy_attachment_returns_same_preparation_without_retirement_or_housekeeping() {
    let array = Array::from_slice(&[1_i32, 2, 3], &[3]);
    array.evaluated().unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap();
    let native = prepared.native.as_ref().unwrap().0;
    let rust = std::ptr::from_ref(&**prepared.node.as_ref().unwrap());
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        done_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    runtime_lock::register_housekeeping_hook(hook);
    let _hook = Unhook;
    HOOKS.with(|x| x.set(0));
    let result = prepared.try_attach(&array);
    done_tx.send(()).unwrap();
    worker.join().unwrap();
    let error = result.unwrap_err();
    assert!(matches!(
        error.cause(),
        PreparedAllocationOwnerCause::RuntimeBusy
    ));
    assert_eq!(error.owner().native.as_ref().unwrap().0, native);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()),
        rust
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(HOOKS.with(Cell::get), 0);
    error.into_parts().1.try_attach(&array).unwrap();
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(_hook);
    drop(array);
    wait(&drops, 1, &stream());
}

#[test]
fn lazy_and_empty_backing_preserve_owner_and_actual_native_cause() {
    let stream = stream();
    let input = Array::from_slice(&[2_i32, 3], &[2]);
    let lazy = input.square(&stream).unwrap();
    let error = PreparedAllocationOwner::try_new(17_usize)
        .unwrap()
        .try_attach(&lazy)
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        PreparedAllocationOwnerCause::UncertifiedBacking
    ));
    let prepared = error.into_parts().1;
    assert_eq!(*prepared.owner(), 17);
    // Real C argument validation supplies the original native exception; the
    // private test seam deliberately omits the callback, leaving both nodes idle.
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
    match error.cause() {
        PreparedAllocationOwnerCause::Native(error) => assert!(error
            .to_string()
            .contains("Invalid prepared allocation owner")),
        other => panic!("unexpected cause {other:?}"),
    }
    assert_eq!(error.into_parts().1.into_owner(), 17);

    // Logical emptiness does not imply absent physical backing. The CPU
    // allocator can retain a real zero-capacity allocation; Metal returns null.
    let empty = Array::from_slice::<i32>(&[], &[0]);
    empty.evaluated().unwrap();
    assert_eq!(empty.size(), 0);
    assert_eq!(empty.nbytes(), 0);
    // Some means the facts are known, including the allocation-free zero
    // sentinel. Only a nonzero physical identity can receive this owner.
    let info = empty
        .allocation_info()
        .unwrap()
        .expect("available empty facts are known");
    assert!(
        info.bytes() == 0 || info.bytes() == std::mem::size_of::<usize>(),
        "empty CPU backing includes its allocation header"
    );
    let no_allocation = crate::AllocationInfo::from_native(
        0,
        0,
        safemlx_sys::mlx_memory_placement {
            kind: 1,
            device: -1,
            device_count: 0,
        },
    )
    .identity();
    let alias = empty.clone();
    let drops = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap();
    let native = prepared.native.as_ref().unwrap().0;
    let rust = std::ptr::from_ref(&**prepared.node.as_ref().unwrap());
    if info.identity() != no_allocation {
        prepared.try_attach(&empty).unwrap();
        assert_eq!(empty.allocation_info().unwrap(), Some(info));
        assert_eq!(alias.allocation_info().unwrap(), Some(info));
        drop(empty);
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(alias);
        wait(&drops, 1, &stream);
    } else {
        let error = prepared.try_attach(&empty).unwrap_err();
        assert!(matches!(
            error.cause(),
            PreparedAllocationOwnerCause::NoAllocation
        ));
        assert!(Arc::ptr_eq(&error.owner().owner().0, &drops));
        assert_eq!(error.owner().native.as_ref().unwrap().0, native);
        assert_eq!(
            std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()),
            rust
        );
        assert_eq!(empty.allocation_info().unwrap(), Some(info));
        assert_eq!(alias.allocation_info().unwrap(), Some(info));
        drop((empty, alias));
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(error);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn mixed_legacy_and_prepared_owners_follow_preexisting_views_and_lazy_aliases() {
    let stream = stream();
    let input = Array::from_slice(&[1_i32, 2, 3, 4], &[4]);
    let view = input.try_index_device(1.., &stream).unwrap();
    view.evaluated().unwrap();
    let identity = input.allocation_info().unwrap();
    assert_eq!(view.allocation_info().unwrap(), identity);
    let drops = Arc::new(AtomicUsize::new(0));
    input.retain_allocation_owner(Probe(drops.clone())).unwrap();
    for _ in 0..64 {
        PreparedAllocationOwner::try_new(Probe(drops.clone()))
            .unwrap()
            .try_attach(&view)
            .unwrap();
    }
    assert_eq!(input.allocation_info().unwrap(), identity);
    let lazy = view.square(&stream).unwrap();
    drop((view, input));
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(lazy.evaluated().unwrap().as_slice::<i32>(), &[4, 9, 16]);
    drop(lazy);
    wait(&drops, 65, &stream);
}

#[test]
fn host_prepared_owners_follow_shared_backing_or_independent_copies() {
    let devices = if cfg!(feature = "metal") {
        vec![DeviceType::Cpu, DeviceType::Gpu]
    } else {
        vec![DeviceType::Cpu]
    };
    for device in devices {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut buffer =
            HostTransferBuffer::new(&[2], Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
        buffer.as_bytes_mut().unwrap().copy_from_slice(
            &[4_i32, 9]
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>(),
        );
        let buffer = Arc::new(buffer.freeze());
        let alias = buffer.clone();
        let array = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        array.evaluated().unwrap();
        let host = buffer.allocation_info().unwrap();
        let same = array.allocation_info().unwrap() == Some(host);
        let drops = Arc::new(AtomicUsize::new(0));
        buffer
            .try_attach_prepared_allocation_owner(
                PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap(),
            )
            .unwrap();
        array.retain_allocation_owner(Probe(drops.clone())).unwrap();
        PreparedAllocationOwner::try_new(Probe(drops.clone()))
            .unwrap()
            .try_attach(&array)
            .unwrap();
        drop(buffer);
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(alias);
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), usize::from(!same));
        assert_eq!(array.evaluated().unwrap().as_slice::<i32>(), &[4, 9]);
        drop(array);
        wait(&drops, 3, &stream);
    }
}

#[test]
fn failed_or_unwound_later_handoff_does_not_release_the_attached_prefix() {
    let stream = stream();
    let first = Array::from_slice(&[5_i32], &[1]);
    first.evaluated().unwrap();
    let lazy = first.square(&stream).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    PreparedAllocationOwner::try_new(Probe(drops.clone()))
        .unwrap()
        .try_attach(&first)
        .unwrap();
    let pending = PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap();
    let failure = pending.try_attach(&lazy).unwrap_err();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        failure
            .into_parts()
            .1
            .attach_with(|_, _, _| std::panic::panic_any(91_u32))
    }))
    .unwrap_err();
    assert_eq!(*panic.downcast::<u32>().unwrap(), 91);
    assert_eq!(drops.load(Ordering::SeqCst), 1); // guard dropped before pending owner
    drop(first);
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(lazy);
    wait(&drops, 2, &stream);
}

#[test]
fn cold_preparation_initializes_once_before_bounded_handoff() {
    const CHILD: &str = "SAFEMLX_PREPARED_OWNER_COLD_INITIALIZATION_CHILD";
    const VERIFIED: &str = "cold prepared-owner initialization verified";
    const NAME: &str = "allocation_retention::prepared::tests::cold_preparation_initializes_once_before_bounded_handoff";
    if std::env::var_os(CHILD).is_none() {
        // A new test process makes the uninitialized state real, independent of
        // other tests that already used the global native error handler.
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--test-threads=1", "--nocapture"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "cold child failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains(VERIFIED),
            "cold child did not execute the exact test"
        );
        return;
    }
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (false, 0));
    let facts = PreparedAllocationOwner::<usize>::layout();
    assert!(facts.native_node_bytes() > 0);
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (false, 0));
    let input = Array::from_int(7);
    // This real constructor does not initialize the guarded status converter.
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (false, 0));
    runtime_lock::register_housekeeping_hook(hook);
    let _hook = Unhook;
    HOOKS.with(|x| x.set(0));
    let prepared = PreparedAllocationOwner::try_new(17_usize).unwrap();
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (true, 1));
    // A real rejected C operation proves the same original native diagnostic is
    // available without entering Once at the bounded attachment boundary.
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
    match error.cause() {
        PreparedAllocationOwnerCause::Native(error) => assert!(error
            .to_string()
            .contains("Invalid prepared allocation owner")),
        other => panic!("unexpected cause {other:?}"),
    }
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (true, 1));
    assert_eq!(*error.owner().owner(), 17);
    error.into_parts().1.try_attach(&input).unwrap();
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), (true, 1));
    assert_eq!(HOOKS.with(Cell::get), 0);
    drop(_hook);
    drop(input);
    reclaim_allocation_owners();
    println!("{VERIFIED}");
}

mod deallocation;

#[test]
fn original_attachment_busy_and_unknown_preserve_both_actual_prepared_nodes() {
    // This is a real ordinary allocation, never a fabricated original birth.
    // Its genuine descriptor supplies inputs for the private handoff's refusal
    // paths; production obtains those inputs only from a checked original witness.
    let source = Array::from_slice(&[17.0f32, 19.0, 23.0], &[3]);
    let facts = {
        let _guard = runtime_lock::enter();
        let mut descriptor = std::mem::MaybeUninit::<safemlx_sys::mlx_array_descriptor>::uninit();
        // SAFETY: valid borrowed source and exact output representation.
        assert_eq!(
            unsafe {
                safemlx_sys::mlx_array_descriptor_read(descriptor.as_mut_ptr(), source.as_ptr())
            },
            0
        );
        let descriptor = unsafe { descriptor.assume_init() };
        assert!(descriptor.known && descriptor.identity != 0);
        safemlx_sys::mlx_original_buffer_info {
            host_control_bytes: 0,
            known: descriptor.known,
            identity: descriptor.identity,
            charged_bytes: descriptor.allocation_bytes,
            placement: descriptor.placement,
        }
    };
    let drops = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedAllocationOwner::try_new(Probe(drops.clone())).unwrap();
    let native = prepared.native.as_ref().unwrap().0.as_ptr();
    let rust = std::ptr::from_ref(&**prepared.node.as_ref().unwrap());
    let (ready_tx, ready_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        done_rx.recv_timeout(Duration::from_secs(10)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(10)).is_ok();
    let result = prepared.attach_original(&source, facts, None);
    let released = done_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), OriginalBufferCause::RuntimeBusy);
    assert_eq!(error.owner().native.as_ref().unwrap().0.as_ptr(), native);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()),
        rust
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let prepared = error.into_parts().1;
    let ordinary = crate::OrdinarySubmissionEntry::enter().unwrap();
    let error = prepared.attach_original(&source, facts, None).unwrap_err();
    assert_eq!(error.cause(), OriginalBufferCause::UncertifiedBacking);
    assert_eq!(error.owner().native.as_ref().unwrap().0.as_ptr(), native);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()),
        rust
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(ordinary);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
