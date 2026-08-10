use safemlx::{
    Array, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy,
    HostTransferStorageKind, Stream,
};

#[test]
fn host_transfer_round_trip_preserves_metadata_and_values() {
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
        HostTransferStorageKind::Cpu | HostTransferStorageKind::MetalShared
    ));

    let values = host
        .as_bytes()
        .unwrap()
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
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
fn noncontiguous_sources_and_empty_buffers_preserve_logical_geometry() {
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
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![1, 4, 2, 5, 3, 6]);

    let empty =
        HostTransferBuffer::new(&[0, 3], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
    assert!(empty.is_empty().unwrap());
    assert_eq!(empty.nbytes().unwrap(), 0);
    assert!(empty.as_bytes().unwrap().is_empty());
}

#[test]
fn managed_policy_is_explicitly_rejected_without_cuda() {
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
    use safemlx::EventBackend;

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
