use super::*;
use crate::{
    Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy, Stream, SubmissionScope,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Probe(Arc<AtomicUsize>);

impl Drop for Probe {
    fn drop(&mut self) {
        assert!(crate::can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
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
    stream.synchronize().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while drops.load(Ordering::SeqCst) != expected {
        crate::memory::clear_cache().unwrap();
        reclaim_allocation_owners();
        assert!(std::time::Instant::now() < deadline, "owner did not retire");
        std::thread::yield_now();
    }
}

#[test]
fn deferred_owner_is_cold_and_follows_preexisting_lazy_clones_and_strided_views() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let root = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6, 7, 8], &[8])
            .square(&stream)
            .unwrap();
        let clone = root.clone();
        let view = root.as_strided(&[4][..], &[2][..], 1, &stream).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        root.retain_deferred_allocation_owner(Probe(drops.clone()))
            .unwrap();
        assert_eq!(root.allocation_info().unwrap(), None);
        assert_eq!(clone.allocation_info().unwrap(), None);
        assert_eq!(view.allocation_info().unwrap(), None);
        drop(root);
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);

        view.evaluated().unwrap();
        let allocation = view.allocation_info().unwrap().unwrap();
        assert_eq!(clone.allocation_info().unwrap(), Some(allocation));
        let later_view = clone.as_strided(&[2][..], &[2][..], 0, &stream).unwrap();
        later_view.evaluated().unwrap();
        assert_eq!(later_view.allocation_info().unwrap(), Some(allocation));
        assert_eq!(
            view.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            vec![4, 16, 36, 64]
        );
        drop((clone, view));
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(later_view);
        reclaim_until(&drops, 1, &stream);
    }
}

#[test]
fn deferred_owner_retires_without_materializing_an_abandoned_graph() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_f32, 3.], &[2]).exp(&stream).unwrap();
    let clone = root.clone();
    let drops = Arc::new(AtomicUsize::new(0));
    root.retain_deferred_allocation_owner(Probe(drops.clone()))
        .unwrap();
    drop(root);
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(clone.allocation_info().unwrap(), None);
    drop(clone);
    reclaim_until(&drops, 1, &stream);
}

#[test]
fn completed_only_attachment_still_rejects_lazy_values_and_preserves_owner_for_deferred_use() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_i32, 3], &[2])
        .square(&stream)
        .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let failure = root
        .retain_allocation_owner(Probe(drops.clone()))
        .unwrap_err();
    assert_eq!(root.allocation_info().unwrap(), None);
    let (_, owner) = failure.into_parts();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    root.retain_deferred_allocation_owner(owner).unwrap();
    assert_eq!(root.allocation_info().unwrap(), None);
    drop(root);
    reclaim_until(&drops, 1, &stream);
}

#[test]
fn deferred_output_authority_survives_materialization_and_late_attachment_to_aliases() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let input = Array::from_slice(&(1..65).map(|v| v as f32).collect::<Vec<_>>(), &[64]);
        let output = input.square(&stream).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        output
            .retain_deferred_allocation_owner(Probe(drops.clone()))
            .unwrap();
        drop(input); // The selected unary implementation may donate its backing.
        assert_eq!(
            output.evaluated().unwrap().as_slice::<f32>(),
            (1..65).map(|v| (v * v) as f32).collect::<Vec<_>>()
        );
        let allocation = output.allocation_info().unwrap().unwrap();
        let alias = output.as_strided(&[32][..], &[2][..], 0, &stream).unwrap();
        alias.evaluated().unwrap();
        output
            .retain_deferred_allocation_owner(Probe(drops.clone()))
            .unwrap();
        assert_eq!(output.allocation_info().unwrap(), Some(allocation));
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation));
        drop(output);
        reclaim_allocation_owners();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(alias);
        reclaim_until(&drops, 2, &stream);
    }
}

#[test]
fn deferred_owner_outlives_async_submission_without_public_array_handles() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut scope = SubmissionScope::begin().unwrap();
        let output = Array::from_slice(&vec![1.25_f32; 1 << 18], &[1 << 18])
            .square(&stream)
            .unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        output
            .retain_deferred_allocation_owner(Probe(drops.clone()))
            .unwrap();
        let event = crate::transforms::async_eval_with_event([&output]).unwrap();
        scope.seal();
        drop(output);
        reclaim_allocation_owners();
        if drops.load(Ordering::SeqCst) != 0 {
            assert!(scope.status().is_settled());
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
fn deferred_view_authority_reaches_shared_host_storage_and_separate_native_aliases() {
    for device in devices() {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut buffer =
            HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        let values = [1_f32, 2., 3., 4.];
        let bytes = values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>();
        buffer.as_bytes_mut().unwrap().copy_from_slice(&bytes);
        let buffer = buffer.freeze();
        let host = buffer.allocation_info().unwrap();
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
        // The transfer event covers the copy, but its returned descriptors still
        // need their own evaluation boundary before cold backing certification.
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        let native = first.allocation_info().unwrap().unwrap();
        if cfg!(feature = "metal") && device == DeviceType::Gpu {
            assert_eq!(native, host, "exercise shared host/native backing");
            assert_eq!(second.allocation_info().unwrap(), Some(host));
        }
        let view = first.as_strided(&[2][..], &[2][..], 1, &stream).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        view.retain_deferred_allocation_owner(Probe(drops.clone()))
            .unwrap();
        assert_eq!(view.allocation_info().unwrap(), None);
        assert_eq!(
            view.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
            vec![2., 4.]
        );
        assert_eq!(view.allocation_info().unwrap(), Some(native));
        drop((view, first));
        if native == host {
            reclaim_allocation_owners();
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(second);
            reclaim_allocation_owners();
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            drop(buffer);
        } else {
            drop((second, buffer));
        }
        reclaim_until(&drops, 1, &stream);
    }
}
