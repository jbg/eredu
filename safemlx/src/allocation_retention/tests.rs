use super::*;
use crate::{
    ops::indexing::TryIndexOp, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy,
    Stream, SubmissionScope,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

struct Probe {
    drops: Arc<AtomicUsize>,
    thread: Option<Arc<Mutex<Option<std::thread::ThreadId>>>>,
}

impl Probe {
    fn new(drops: &Arc<AtomicUsize>) -> Self {
        Self {
            drops: drops.clone(),
            thread: None,
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        assert!(
            crate::can_reclaim_submission_resources(),
            "owner dropped under native lock"
        );
        if let Some(thread) = &self.thread {
            *thread.lock().unwrap() = Some(std::thread::current().id());
        }
        // Reentrant native entry during ordinary retirement must remain safe.
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        stream.synchronize().unwrap();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn devices() -> Vec<DeviceType> {
    if cfg!(feature = "metal") {
        vec![DeviceType::Cpu, DeviceType::Gpu]
    } else {
        vec![DeviceType::Cpu]
    }
}

fn reclaim_until(drops: &Arc<AtomicUsize>, expected: usize, stream: &Stream) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while drops.load(Ordering::SeqCst) != expected {
        stream.synchronize().unwrap();
        reclaim_allocation_owners();
        assert!(
            std::time::Instant::now() < deadline,
            "allocation owner did not retire"
        );
        std::thread::yield_now();
    }
}

#[test]
fn allocation_retention_preserves_preexisting_views_and_lazy_children() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let root = Array::from_slice(&(0..32).collect::<Vec<i32>>(), &[4, 8]);
        let view = root.try_index_device((2.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        let identity = root.allocation_info().unwrap().unwrap();
        assert_eq!(view.allocation_info().unwrap(), Some(identity));
        let drops = Arc::new(AtomicUsize::new(0));
        root.retain_allocation_owner(Probe::new(&drops)).unwrap();
        view.retain_allocation_owner(Probe::new(&drops)).unwrap();
        assert_eq!(root.allocation_info().unwrap(), Some(identity));
        assert_eq!(view.allocation_info().unwrap(), Some(identity));
        let alias = root.clone();
        let lazy = view.square(&stream).unwrap();
        drop((root, view));
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(
            alias.evaluated().unwrap().as_slice::<i32>(),
            (0..32).collect::<Vec<_>>()
        );
        drop(alias);
        reclaim_allocation_owners();
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "lazy child retains its input backing"
        );
        drop(lazy);
        reclaim_until(&drops, 2, &stream);
    }
}

#[test]
fn allocation_retention_host_storage_includes_independently_wrapped_aliases() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut buffer =
            HostTransferBuffer::new(&[4, 8], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        let values = (0..32).map(|v| v as f32 * 0.25).collect::<Vec<_>>();
        let bytes = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        buffer.as_bytes_mut().unwrap().copy_from_slice(&bytes);
        let buffer = buffer.freeze();
        let first = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let second = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        let host = buffer.allocation_info().unwrap();
        let native = first.allocation_info().unwrap().unwrap();
        if cfg!(feature = "metal") && device == DeviceType::Gpu {
            assert_eq!(native, host, "Metal must exercise shared host backing");
        }
        let drops = Arc::new(AtomicUsize::new(0));
        first.retain_allocation_owner(Probe::new(&drops)).unwrap();
        assert_eq!(first.allocation_info().unwrap(), Some(native));
        let view = second.try_index_device((2.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &values[16..]);
        drop(first);
        reclaim_allocation_owners();
        if native == host {
            assert_eq!(second.allocation_info().unwrap(), Some(host));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop((second, view));
            reclaim_allocation_owners();
            assert_eq!(
                drops.load(Ordering::SeqCst),
                0,
                "host buffer retains the same backing"
            );
            drop(buffer);
        } else {
            // Copying host transfers own independent device allocations.
            assert_ne!(
                second.allocation_info().unwrap().unwrap().identity(),
                native.identity()
            );
            drop((second, view, buffer));
        }
        reclaim_until(&drops, 1, &stream);
    }
}

#[test]
fn allocation_retention_follows_actual_donated_or_independent_result_storage() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let root = Array::from_slice(&(1..65).map(|n| n as f32).collect::<Vec<_>>(), &[64]);
        let identity = root.allocation_info().unwrap().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        root.retain_allocation_owner(Probe::new(&drops)).unwrap();
        let output = root.square(&stream).unwrap();
        drop(root);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<f32>(),
            (1..65).map(|n| (n * n) as f32).collect::<Vec<_>>()
        );
        if output.allocation_info().unwrap() == Some(identity) {
            reclaim_allocation_owners();
            assert_eq!(
                drops.load(Ordering::SeqCst),
                0,
                "donation preserves allocation owners"
            );
        }
        drop(output);
        reclaim_until(&drops, 1, &stream);
    }
}

#[test]
fn allocation_retention_outlives_async_submission_and_dropped_public_handles() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut scope = SubmissionScope::begin().unwrap();
        let root = Array::from_slice(&vec![1.25_f32; 1 << 18], &[1 << 18]);
        let drops = Arc::new(AtomicUsize::new(0));
        root.retain_allocation_owner(Probe::new(&drops)).unwrap();
        let output = root.square(&stream).unwrap();
        let event = crate::transforms::async_eval_with_event([&output]).unwrap();
        scope.seal();
        drop((root, output));
        // Submission pins remain physical owners even without public arrays.
        reclaim_allocation_owners();
        if drops.load(Ordering::SeqCst) != 0 {
            assert!(
                scope.status().is_settled(),
                "unsettled work released its physical owner"
            );
        }
        event.synchronize().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !scope.progress().is_settled() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        drop((event, scope));
        reclaim_until(&drops, 1, &stream);
    }
}

#[test]
fn allocation_retention_rejects_unfinished_backing_without_consuming_owner() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[1_i32, 2, 3, 4], &[4]);
    let lazy = root.square(&stream).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let error = lazy
        .retain_allocation_owner(Probe::new(&drops))
        .unwrap_err();
    assert_eq!(
        lazy.allocation_info().unwrap(),
        None,
        "attachment must not evaluate"
    );
    let (_, owner) = error.into_parts();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    lazy.evaluated().unwrap();
    lazy.retain_allocation_owner(owner).unwrap();
    drop((root, lazy));
    reclaim_until(&drops, 1, &stream);
}

#[test]
fn allocation_retention_native_argument_error_preserves_payload_ownership() {
    let root = Array::from_slice(&[1_i32, 2, 3, 4], &[4]);
    let drops = Arc::new(AtomicUsize::new(0));
    let mut node = Box::new(OwnedNode {
        retired: RetiredOwner {
            next: ptr::null_mut(),
            destroy: destroy::<Probe>,
        },
        owner: Probe::new(&drops),
    });
    let mut attached = false;
    let result = <() as Guarded>::try_from_op(|_| unsafe {
        // SAFETY: all pointers are valid; a missing callback is intentionally
        // invalid and must fail before native ownership transfers.
        safemlx_sys::mlx_array_retain_allocation_owner(
            &mut attached,
            root.as_ptr(),
            ptr::from_mut(&mut node.retired).cast(),
            None,
        )
    });
    assert!(result.is_err());
    assert!(!attached);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(node);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn allocation_retention_native_drop_only_queues_owners_for_unlocked_host_reclamation() {
    let root = Array::from_slice(&[1_i32, 2, 3, 4], &[4]);
    let drops = Arc::new(AtomicUsize::new(0));
    let dropped_on = Arc::new(Mutex::new(None));
    root.retain_allocation_owner(Probe {
        drops: drops.clone(),
        thread: Some(dropped_on.clone()),
    })
    .unwrap();
    let drop_thread = std::thread::spawn(move || {
        let id = std::thread::current().id();
        drop(root);
        id
    })
    .join()
    .unwrap();
    {
        let _lock = runtime_lock::enter();
        assert_eq!(
            reclaim_allocation_owners(),
            0,
            "reclamation cannot run under the native lock"
        );
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    reclaim_until(&drops, 1, &stream);
    assert_ne!(*dropped_on.lock().unwrap(), Some(drop_thread));
}

#[test]
fn allocation_retention_host_first_attachment_survives_native_aliases() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut buffer =
            HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        let values = [1.25_f32, 2.5, 3.75, 5.0];
        let bytes = values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>();
        buffer.as_bytes_mut().unwrap().copy_from_slice(&bytes);
        let buffer = buffer.freeze();
        let info = buffer.allocation_info().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        // No native array exists when the host storage acquires its owner.
        buffer.retain_allocation_owner(Probe::new(&drops)).unwrap();
        assert_eq!(buffer.allocation_info().unwrap(), info);
        let first = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let second = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        let native = first.allocation_info().unwrap().unwrap();
        if cfg!(feature = "metal") && device == DeviceType::Gpu {
            assert_eq!(native, info);
        }
        drop(buffer);
        reclaim_allocation_owners();
        if native == info {
            assert_eq!(drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(second.evaluated().unwrap().as_slice::<f32>(), values);
        drop((first, second));
        reclaim_until(&drops, 1, &stream);
    }
}

#[test]
fn allocation_retention_generations_do_not_reuse_retired_storage_keys() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut retired_keys = std::collections::BTreeSet::new();
    let drops = Arc::new(AtomicUsize::new(0));
    for _ in 0..64 {
        let array = Array::from_slice(&[1_u32, 2, 3, 4], &[4]);
        let info = array.allocation_info().unwrap().unwrap();
        assert!(
            retired_keys.insert(info.identity()),
            "native generation was reused"
        );
        array.retain_allocation_owner(Probe::new(&drops)).unwrap();
        drop(array);
        let buffer = HostTransferBuffer::new(&[4], Dtype::Uint32, HostTransferPolicy::Transfer)
            .unwrap()
            .freeze();
        let info = buffer.allocation_info().unwrap();
        assert!(
            retired_keys.insert(info.identity()),
            "host generation was reused"
        );
        buffer.retain_allocation_owner(Probe::new(&drops)).unwrap();
        drop(buffer);
    }
    reclaim_until(&drops, 128, &stream);
    assert_eq!(retired_keys.len(), 128);
}
