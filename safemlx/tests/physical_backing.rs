//! The native cache owns the same physical record before and after array reuse.
use safemlx::{
    AllocationInfo, Array, PhysicalBackingCustody, PhysicalBackingObserver,
    PhysicalBackingPublication,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Barrier,
};

#[derive(Debug)]
struct Refused;
impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("fixture physical capacity exhausted")
    }
}
impl std::error::Error for Refused {}
struct Observer;
static OBSERVER: Observer = Observer;
static REFUSE: AtomicBool = AtomicBool::new(false);
static ROOTS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CONTROLS: AtomicUsize = AtomicUsize::new(0);
static RESERVED: AtomicUsize = AtomicUsize::new(0);
static REGISTERED: AtomicUsize = AtomicUsize::new(0);
#[derive(Debug)]
struct Charge {
    facts: AllocationInfo,
    published: bool,
}
impl PhysicalBackingPublication for Charge {
    fn publish(&mut self) -> safemlx::error::Result<()> {
        assert!(!self.published);
        assert!(RESERVED.fetch_sub(self.facts.bytes(), Ordering::SeqCst) >= self.facts.bytes());
        REGISTERED.fetch_add(self.facts.bytes(), Ordering::SeqCst);
        self.published = true;
        Ok(())
    }
}
impl Drop for Charge {
    fn drop(&mut self) {
        ROOTS.fetch_sub(1, Ordering::SeqCst);
        BYTES.fetch_sub(self.facts.bytes(), Ordering::SeqCst);
        CONTROLS.fetch_sub(self.facts.host_control_bytes(), Ordering::SeqCst);
        if self.published {
            REGISTERED.fetch_sub(self.facts.bytes(), Ordering::SeqCst);
        } else {
            RESERVED.fetch_sub(self.facts.bytes(), Ordering::SeqCst);
        }
    }
}
impl PhysicalBackingObserver for Observer {
    fn admit(&self, facts: AllocationInfo) -> safemlx::error::Result<PhysicalBackingCustody> {
        if REFUSE.load(Ordering::SeqCst) {
            return Err(safemlx::error::Exception::from_source(Refused));
        }
        ROOTS.fetch_add(1, Ordering::SeqCst);
        BYTES.fetch_add(facts.bytes(), Ordering::SeqCst);
        CONTROLS.fetch_add(facts.host_control_bytes(), Ordering::SeqCst);
        RESERVED.fetch_add(facts.bytes(), Ordering::SeqCst);
        Ok(PhysicalBackingCustody::new_pending(Charge {
            facts,
            published: false,
        }))
    }
}
fn clear() {
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
}
// The callback queue may be drained by a different host thread immediately
// after native retirement. Aliases preserve the actual root while that drain
// runs, and final destruction delivers its charge exactly once.
fn concurrent_alias_retirement() {
    let previous = safemlx::memory::set_cache_limit(0).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = stop.clone();
    let reclaimer = std::thread::spawn(move || {
        while !worker_stop.load(Ordering::Acquire) {
            safemlx::reclaim_allocation_owners();
            std::thread::yield_now();
        }
    });
    for _ in 0..32 {
        let source = Array::try_from_slice(&[3.0f32, -5.0, 7.0, 11.0], &[4]).unwrap();
        let facts = source.allocation_info().unwrap().unwrap();
        let alias = source.try_clone_handle().unwrap();
        let retained = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let worker_retained = retained.clone();
        let worker_release = release.clone();
        let holder = std::thread::spawn(move || {
            worker_retained.wait();
            worker_release.wait();
            drop(alias);
        });
        drop(source);
        clear();
        retained.wait();
        assert_eq!(ROOTS.load(Ordering::SeqCst), 1);
        assert_eq!(BYTES.load(Ordering::SeqCst), facts.bytes());
        assert_eq!(CONTROLS.load(Ordering::SeqCst), facts.host_control_bytes());
        release.wait();
        holder.join().unwrap();
        clear();
        // A concurrent drain may already own the retired queue node.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while ROOTS.load(Ordering::SeqCst) != 0
            || CONTROLS.load(Ordering::SeqCst) != 0
            || BYTES.load(Ordering::SeqCst) != 0
            || REGISTERED.load(Ordering::SeqCst) != 0
            || RESERVED.load(Ordering::SeqCst) != 0
        {
            assert!(std::time::Instant::now() < deadline);
            safemlx::reclaim_allocation_owners();
            std::thread::yield_now();
        }
        assert_eq!(BYTES.load(Ordering::SeqCst), 0);
        assert_eq!(REGISTERED.load(Ordering::SeqCst), 0);
        assert_eq!(RESERVED.load(Ordering::SeqCst), 0);
    }
    stop.store(true, Ordering::Release);
    reclaimer.join().unwrap();
    safemlx::memory::set_cache_limit(previous).unwrap();
}

#[cfg(feature = "cuda")]
fn empty_arrays_cross_streams_without_a_physical_root() {
    let gpu = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let cpu = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let empty = Array::try_from_slice::<f32>(&[], &[0]).unwrap();
    let device = empty.copy(&gpu).unwrap();
    let host = device.copy(&cpu).unwrap();
    let total = host.sum(false, &gpu).unwrap();
    let mut value = [1.0f32];
    total
        .evaluated()
        .unwrap()
        .try_copy_into(&mut value)
        .unwrap();
    assert_eq!(value, [0.0]);
    assert!(device.allocation_info().unwrap().is_none());
    assert!(host.allocation_info().unwrap().is_none());
    drop((total, host, device, empty));
    gpu.synchronize().unwrap();
    cpu.synchronize().unwrap();
    clear();
    assert_eq!(ROOTS.load(Ordering::SeqCst), 0);
    assert_eq!(CONTROLS.load(Ordering::SeqCst), 0);
}

#[test]
fn cached_backing_retains_identity_charge_and_checked_admission() {
    clear();
    let previous = safemlx::memory::set_cache_limit(1 << 20).unwrap();
    // Existing allocations are included when the observer is first installed.
    let existing = Array::try_from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[4]).unwrap();
    let before = existing.allocation_info().unwrap().unwrap();
    assert!(before.host_control_bytes() >= safemlx::physical_backing_control_bytes());
    safemlx::observe_physical_backings(&OBSERVER).unwrap();
    assert_eq!(ROOTS.load(Ordering::SeqCst), 1);
    assert_eq!(BYTES.load(Ordering::SeqCst), before.bytes());
    assert_eq!(REGISTERED.load(Ordering::SeqCst), before.bytes());
    assert_eq!(RESERVED.load(Ordering::SeqCst), 0);
    drop(existing);
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        ROOTS.load(Ordering::SeqCst),
        1,
        "cache retains physical charge"
    );
    let reused = Array::try_from_slice(&[11.0f32, -13.0, 17.0, 19.0], &[4]).unwrap();
    assert_eq!(reused.allocation_info().unwrap(), Some(before));
    let independent = Array::try_from_slice(&[23.0f32, 29.0, -31.0, 37.0], &[4]).unwrap();
    assert_ne!(
        independent.allocation_info().unwrap().unwrap().identity(),
        before.identity()
    );
    assert_eq!(ROOTS.load(Ordering::SeqCst), 2);
    assert_eq!(
        reused.evaluated().unwrap().as_slice::<f32>(),
        &[11.0, -13.0, 17.0, 19.0]
    );
    drop((reused, independent));
    clear();
    assert_eq!(ROOTS.load(Ordering::SeqCst), 0);
    assert_eq!(BYTES.load(Ordering::SeqCst), 0);
    assert_eq!(CONTROLS.load(Ordering::SeqCst), 0);
    assert_eq!(REGISTERED.load(Ordering::SeqCst), 0);
    assert_eq!(RESERVED.load(Ordering::SeqCst), 0);
    // Ordinary transfer storage and every Array view expose the same root key.
    let mut transfer = safemlx::HostTransferBuffer::new(
        &[4],
        safemlx::Dtype::Float32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap();
    for (slot, value) in transfer
        .as_bytes_mut()
        .unwrap()
        .chunks_exact_mut(4)
        .zip([2.0f32, -3.0, 5.0, 7.0])
    {
        slot.copy_from_slice(&value.to_ne_bytes());
    }
    let transfer = transfer.freeze();
    let host = transfer.allocation_info().unwrap();
    assert!(host.host_control_bytes() > safemlx::physical_backing_control_bytes());
    assert_eq!(ROOTS.load(Ordering::SeqCst), 1);
    assert_eq!(BYTES.load(Ordering::SeqCst), host.bytes());
    let devices = safemlx::physical_memory_topology().unwrap();
    let device = devices.first().map_or(
        safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
        |device| safemlx::Device::new(safemlx::DeviceType::Gpu, device.ordinal as i32),
    );
    let stream = safemlx::Stream::new_with_device(&device);
    let destination = transfer
        .copy_to_array(&stream)
        .unwrap()
        .synchronize()
        .unwrap();
    // Establish the array's completed descriptor after the transfer event.
    // Cold allocation inspection deliberately does not poll or detach events.
    let _ = destination.evaluated().unwrap();
    let destination_facts = destination.allocation_info().unwrap().unwrap();
    if destination_facts.identity() == host.identity() {
        assert_eq!(destination_facts, host);
        assert_eq!(
            ROOTS.load(Ordering::SeqCst),
            1,
            "zero-copy views share one charged physical root"
        );
    } else {
        assert_eq!(
            ROOTS.load(Ordering::SeqCst),
            2,
            "an independent destination has an independent root"
        );
    }
    drop(transfer);
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        ROOTS.load(Ordering::SeqCst),
        1,
        "completed independent staging retires before its destination"
    );
    let mut output = [0.0f32; 4];
    destination
        .evaluated()
        .unwrap()
        .try_copy_into(&mut output)
        .unwrap();
    assert_eq!(output, [2.0, -3.0, 5.0, 7.0]);
    drop(destination);
    stream.synchronize().unwrap();
    clear();
    assert_eq!(ROOTS.load(Ordering::SeqCst), 0);
    concurrent_alias_retirement();
    #[cfg(feature = "cuda")]
    empty_arrays_cross_streams_without_a_physical_root();
    REFUSE.store(true, Ordering::SeqCst);
    let error = safemlx::HostTransferBuffer::new(
        &[4],
        safemlx::Dtype::Float32,
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    while cause.source().is_some() {
        cause = cause.source().unwrap();
    }
    assert!(
        cause.downcast_ref::<Refused>().is_some(),
        "host birth preserves its typed rejection before native storage: {error:?}"
    );
    assert_eq!(ROOTS.load(Ordering::SeqCst), 0);

    let error = Array::try_from_slice(&[41.0f32], &[1]).unwrap_err();
    let mut cause: &dyn std::error::Error = &error;
    while cause.source().is_some() {
        cause = cause.source().unwrap();
    }
    assert!(
        cause.downcast_ref::<Refused>().is_some(),
        "typed refusal survives native catch: {error:?}"
    );
    assert_eq!(ROOTS.load(Ordering::SeqCst), 0);
    REFUSE.store(false, Ordering::SeqCst);
    safemlx::memory::set_cache_limit(previous).unwrap();
    clear();
}
