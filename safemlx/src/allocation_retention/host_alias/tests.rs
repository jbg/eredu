use super::*;
use crate::{
    ops::indexing::TryIndexOp, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy,
    Stream,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn host_view_attachment_rejects_stale_generation_and_retires_before_immutable_source() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let values = [1.25f32; 16];
    let mut host =
        HostTransferBuffer::new(&[2, 8], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    for (destination, value) in host.as_bytes_mut().unwrap().chunks_exact_mut(4).zip(values) {
        destination.copy_from_slice(&value.to_ne_bytes());
    }
    let host = host.freeze();
    let first = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
    let independent = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
    // Event completion alone does not detach the Array's pending event. The
    // nonpolling witness requires actual descriptor availability as well.
    first.evaluated().unwrap();
    independent.evaluated().unwrap();
    assert_eq!(
        first.allocation_info().unwrap(),
        Some(host.allocation_info().unwrap())
    );
    assert_eq!(
        independent.allocation_info().unwrap(),
        Some(host.allocation_info().unwrap())
    );
    let first_facts = first.inspect_host_transfer_view().unwrap().unwrap().facts;
    let second_facts = independent
        .inspect_host_transfer_view()
        .unwrap()
        .unwrap()
        .facts;
    assert_eq!(first_facts.backing.identity, second_facts.backing.identity);
    assert_eq!(
        first_facts.backing.charged_bytes,
        second_facts.backing.charged_bytes
    );
    assert_ne!(first_facts.view_identity, second_facts.view_identity);
    let refused_drops = Arc::new(AtomicUsize::new(0));
    let refused = PreparedAllocationOwner::try_new(Probe(refused_drops.clone()))
        .unwrap()
        .attach_host_view(&independent, first_facts)
        .unwrap_err();
    let (cause, original) = refused.into_parts();
    assert_eq!(cause, OriginalBufferCause::BirthChanged);
    assert_eq!(refused_drops.load(Ordering::SeqCst), 0);
    drop(original);
    assert_eq!(refused_drops.load(Ordering::SeqCst), 1);

    let drops = Arc::new(AtomicUsize::new(0));
    let (owner, mut retirement) =
        PreparedAllocationOwner::try_new_with_retirement(Probe(drops.clone())).unwrap();
    first
        .inspect_host_transfer_view()
        .unwrap()
        .unwrap()
        .try_attach(owner)
        .unwrap();
    let clone = first.clone();
    let view = first.try_index_device((1.., ..), &stream).unwrap();
    view.evaluated().unwrap();
    assert_eq!(
        view.inspect_host_transfer_view()
            .unwrap()
            .unwrap()
            .facts
            .view_identity,
        first_facts.view_identity
    );
    drop(first);
    stream.synchronize().unwrap();
    retirement.try_reclaim();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(clone);
    retirement.try_reclaim();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &values[8..]);
    drop(view);
    stream.synchronize().unwrap();
    retirement.try_reclaim();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(
        host.allocation_info().unwrap(),
        AllocationInfo::from_native(
            first_facts.backing.identity,
            first_facts.backing.charged_bytes,
            first_facts.backing.placement
        )
        .with_host_controls(first_facts.backing.host_control_bytes)
    );
    assert_eq!(independent.evaluated().unwrap().as_slice::<f32>(), &values);
}
