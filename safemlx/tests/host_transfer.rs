use safemlx::{
    host_transfer_capacity_upper_bound, host_transfer_memory_stats, memory,
    reset_host_transfer_peak_memory, transforms::async_eval_with_event, Array, Device, DeviceType,
    Dtype, EventBackend, HostTransferBuffer, HostTransferPolicy, HostTransferStorageKind,
    ImmutableHostTransferBuffer, Stream,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

const PEAK_TEST_ELEMENTS: i32 = 2 * 1024 * 1024;

fn runtime_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn host_transfer_round_trip_preserves_metadata_and_values() {
    let _guard = runtime_test_guard();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

    let pending =
        HostTransferBuffer::copy_from_array(&source, HostTransferPolicy::Transfer, &stream)
            .unwrap();
    let host = pending.synchronize().unwrap();

    assert_eq!(host.shape().unwrap(), vec![2, 2]);
    assert_eq!(host.dtype().unwrap(), Dtype::Float32);
    assert_eq!(host.len().unwrap(), 4);
    assert_eq!(host.nbytes().unwrap(), 4 * size_of::<f32>());
    assert!(host.capacity().unwrap() >= host.nbytes().unwrap());
    assert_eq!(host.policy().unwrap(), HostTransferPolicy::Transfer);
    assert!(matches!(
        host.storage_kind().unwrap(),
        HostTransferStorageKind::Cpu
            | HostTransferStorageKind::MetalShared
            | HostTransferStorageKind::CudaPinned
    ));

    let values = host
        .as_bytes()
        .unwrap()
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);

    let pending = host.copy_to_array(&stream).unwrap();
    let (round_trip, host) = pending.synchronize().unwrap();
    assert_eq!(host.shape().unwrap(), vec![2, 2]);
    let evaluated = round_trip.evaluated().unwrap();
    assert_eq!(
        evaluated.try_as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn uninitialized_transfer_buffer_is_typed_and_mutable() {
    let _guard = runtime_test_guard();
    let mut buffer =
        HostTransferBuffer::new(&[2], Dtype::Uint32, HostTransferPolicy::Transfer).unwrap();
    buffer
        .as_bytes_mut()
        .unwrap()
        .copy_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
    assert_eq!(buffer.shape().unwrap(), vec![2]);
    assert_eq!(buffer.dtype().unwrap(), Dtype::Uint32);
    assert!(!buffer.is_empty().unwrap());
}

#[test]
fn physical_capacity_bound_and_native_high_water_track_allocation_lifetime() {
    let _guard = runtime_test_guard();
    let probe =
        HostTransferBuffer::new(&[1], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    let kind = probe.storage_kind().unwrap();
    drop(probe);

    reset_host_transfer_peak_memory(kind).unwrap();
    let baseline = host_transfer_memory_stats(kind).unwrap();
    let buffer =
        HostTransferBuffer::new(&[5000], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    let capacity = buffer.capacity().unwrap();
    assert!(capacity > buffer.nbytes().unwrap());
    assert_eq!(
        host_transfer_capacity_upper_bound(buffer.nbytes().unwrap(), HostTransferPolicy::Transfer)
            .unwrap(),
        capacity
    );
    let live = host_transfer_memory_stats(kind).unwrap();
    assert_eq!(live.active_bytes, baseline.active_bytes + capacity);
    assert_eq!(live.active_allocations, baseline.active_allocations + 1);
    assert!(live.peak_bytes >= live.active_bytes);
    assert!(live.peak_allocations >= live.active_allocations);
    drop(buffer);

    let released = host_transfer_memory_stats(kind).unwrap();
    assert_eq!(released.active_bytes, baseline.active_bytes);
    assert_eq!(released.active_allocations, baseline.active_allocations);
    assert!(released.peak_bytes >= baseline.active_bytes + capacity);
    assert!(released.peak_allocations > baseline.active_allocations);
}

#[test]
fn noncontiguous_sources_and_empty_buffers_preserve_logical_geometry() {
    let _guard = runtime_test_guard();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[1u32, 2, 3, 4, 5, 6], &[2, 3]);
    let transposed = source.transpose_axes(&[1, 0], &stream).unwrap();
    let host =
        HostTransferBuffer::copy_from_array(&transposed, HostTransferPolicy::Transfer, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
    assert_eq!(host.shape().unwrap(), vec![3, 2]);
    let values = host
        .as_bytes()
        .unwrap()
        .as_chunks::<{ size_of::<u32>() }>()
        .0
        .iter()
        .map(|bytes| u32::from_ne_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

    let empty =
        HostTransferBuffer::new(&[0, 3], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    assert!(empty.is_empty().unwrap());
    assert_eq!(empty.nbytes().unwrap(), 0);
    assert!(empty.as_bytes().unwrap().is_empty());
}

#[test]
fn immutable_buffers_are_shareable_and_support_repeated_submissions() {
    let _guard = runtime_test_guard();
    assert_send_sync::<ImmutableHostTransferBuffer>();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[9u32, 10], &[2]);
    let immutable =
        HostTransferBuffer::copy_from_array(&source, HostTransferPolicy::Transfer, &stream)
            .unwrap()
            .synchronize()
            .unwrap()
            .freeze();

    let first = immutable.copy_to_array(&stream).unwrap();
    let second = immutable.copy_to_array(&stream).unwrap();
    drop(immutable);
    let first = first.synchronize().unwrap();
    let second = second.synchronize().unwrap();
    assert_eq!(
        first.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &[9, 10]
    );
    assert_eq!(
        second.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &[9, 10]
    );
}

#[test]
fn managed_policy_is_explicitly_rejected_without_cuda() {
    let _guard = runtime_test_guard();
    if cfg!(feature = "cuda") {
        return;
    }
    let error =
        HostTransferBuffer::new(&[1], Dtype::Float32, HostTransferPolicy::Managed).unwrap_err();
    assert!(error.what().contains("Managed host transfer storage"));
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal host-transfer test; run with --ignored on a Metal host"]
fn metal_transfer_uses_shared_storage_and_metal_completion() {
    let _guard = runtime_test_guard();

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
    let pending =
        HostTransferBuffer::copy_from_array(&source, HostTransferPolicy::Transfer, &stream)
            .unwrap();
    assert_eq!(pending.completion().backend().unwrap(), EventBackend::Metal);

    let host = pending.synchronize().unwrap();
    assert_eq!(
        host.storage_kind().unwrap(),
        HostTransferStorageKind::MetalShared
    );
    let pending = host.copy_to_array(&stream).unwrap();
    assert_eq!(pending.completion().backend().unwrap(), EventBackend::Metal);
    let (round_trip, _) = pending.synchronize().unwrap();
    assert_eq!(
        round_trip
            .evaluated()
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[5.0, 6.0, 7.0, 8.0]
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal host-transfer test; run with --ignored on a Metal host"]
fn metal_transfer_promotes_multidimensional_contiguous_buffers() {
    let _guard = runtime_test_guard();

    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let values = (0..(3 * 5 * 7))
        .map(|value| value as f32 + 0.25)
        .collect::<Vec<_>>();
    let source = Array::from_slice(&values, &[3, 5, 7]);
    let host = HostTransferBuffer::copy_from_array(&source, HostTransferPolicy::Transfer, &stream)
        .unwrap()
        .synchronize()
        .unwrap()
        .freeze();

    let promoted = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
    assert_eq!(promoted.shape(), &[3, 5, 7]);
    assert_eq!(
        promoted.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        values
    );
}

fn assert_transfer_peak_is_bounded(
    device: Device,
    backend: EventBackend,
    storage: HostTransferStorageKind,
    host_storage_is_mlx_accounted: bool,
) {
    let producer = Stream::new_with_device(&device);
    let consumer = Stream::new_with_device(&device);
    assert_ne!(producer.get_index().unwrap(), consumer.get_index().unwrap());

    memory::clear_cache().unwrap();
    let source = Array::ones::<f32>(&[PEAK_TEST_ELEMENTS], &producer).unwrap();
    async_eval_with_event([&source])
        .unwrap()
        .synchronize()
        .unwrap();
    memory::clear_cache().unwrap();
    let source_bytes = source.nbytes();
    let to_host_baseline = memory::active_memory().unwrap();
    memory::reset_peak_memory().unwrap();

    let host =
        HostTransferBuffer::copy_from_array(&source, HostTransferPolicy::Transfer, &producer)
            .unwrap()
            .synchronize()
            .unwrap();
    assert_eq!(host.storage_kind().unwrap(), storage);
    assert_eq!(host.nbytes().unwrap(), source_bytes);
    let host_capacity = host.capacity().unwrap();
    let to_host_extra = memory::peak_memory()
        .unwrap()
        .saturating_sub(to_host_baseline);
    let to_host_limit = if host_storage_is_mlx_accounted {
        // The transfer buffer itself is the one expected full-size allocation.
        // A second full-size staging allocation would exceed this allowance.
        host_capacity
            .saturating_add(source_bytes / 4)
            .saturating_add(1 << 20)
    } else {
        // CUDA pinned host storage is intentionally outside MLX's device
        // allocator counters. Any full-size increase here is hidden staging.
        source_bytes / 4 + (1 << 20)
    };
    assert!(
        to_host_extra <= to_host_limit,
        "{backend:?} device-to-host copy added {to_host_extra} MLX bytes for a \
         {source_bytes}-byte payload (limit {to_host_limit})"
    );

    memory::clear_cache().unwrap();
    let to_device_baseline = memory::active_memory().unwrap();
    memory::reset_peak_memory().unwrap();
    let submitted = host.freeze().copy_to_array(&producer).unwrap();
    assert_eq!(submitted.completion().backend().unwrap(), backend);
    consumer.wait_event(submitted.completion()).unwrap();
    let consumed = submitted
        .value()
        .add(Array::from(1.0f32), &consumer)
        .unwrap();
    async_eval_with_event([&consumed])
        .unwrap()
        .synchronize()
        .unwrap();
    let to_device_extra = memory::peak_memory()
        .unwrap()
        .saturating_sub(to_device_baseline);
    // The returned device array and the elementwise consumer are both live.
    // Allow those two outputs plus sub-payload allocator rounding; a third
    // payload-sized allocation would reveal a copy staging buffer.
    let to_device_limit = source_bytes
        .saturating_mul(2)
        .saturating_add(source_bytes / 4)
        .saturating_add(1 << 20);
    assert!(
        to_device_extra <= to_device_limit,
        "{backend:?} host-to-device copy and one consumer added {to_device_extra} MLX bytes for a \
         {source_bytes}-byte payload (limit {to_device_limit})"
    );
    assert_eq!(
        consumed.evaluated().unwrap().try_as_slice::<f32>().unwrap()[0],
        2.0
    );
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
#[test]
fn cpu_transfer_peak_has_no_hidden_full_size_staging() {
    let _guard = runtime_test_guard();
    assert_transfer_peak_is_bounded(
        Device::new(DeviceType::Cpu, 0),
        EventBackend::Cpu,
        HostTransferStorageKind::Cpu,
        true,
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal allocation test; run with --ignored on a Metal host"]
fn metal_transfer_peak_has_no_hidden_full_size_staging() {
    let _guard = runtime_test_guard();
    assert_transfer_peak_is_bounded(
        Device::new(DeviceType::Gpu, 0),
        EventBackend::Metal,
        HostTransferStorageKind::MetalShared,
        true,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit CUDA allocation test; requires a CUDA-capable host"]
fn cuda_transfer_peak_has_no_hidden_full_size_staging() {
    let _guard = runtime_test_guard();
    assert!(safemlx::cuda::is_available().unwrap());
    assert_transfer_peak_is_bounded(
        Device::new(DeviceType::Gpu, 0),
        EventBackend::Cuda,
        HostTransferStorageKind::CudaPinned,
        false,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit CUDA storage test; requires a CUDA-capable host"]
fn cuda_managed_and_transfer_policies_have_distinct_storage_identity() {
    let _guard = runtime_test_guard();
    assert!(safemlx::cuda::is_available().unwrap());
    let transfer =
        HostTransferBuffer::new(&[1024], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    let managed =
        HostTransferBuffer::new(&[1024], Dtype::Float32, HostTransferPolicy::Managed).unwrap();
    assert_eq!(
        transfer.storage_kind().unwrap(),
        HostTransferStorageKind::CudaPinned
    );
    assert_eq!(
        managed.storage_kind().unwrap(),
        HostTransferStorageKind::CudaManaged
    );
    let pinned = host_transfer_memory_stats(HostTransferStorageKind::CudaPinned).unwrap();
    assert!(pinned.active_bytes >= transfer.capacity().unwrap());
    assert!(pinned.peak_bytes >= pinned.active_bytes);
    let managed_stats = host_transfer_memory_stats(HostTransferStorageKind::CudaManaged).unwrap();
    assert!(managed_stats.active_bytes >= managed.capacity().unwrap());
    assert!(managed_stats.peak_bytes >= managed_stats.active_bytes);
}
