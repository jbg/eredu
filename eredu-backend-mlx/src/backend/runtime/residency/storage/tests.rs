use super::*;
use eredu_checkpoint::store::{CheckpointSource, MemoryWeightStore};
use safemlx::{
    ops::indexing::TryIndexOp, Device, DeviceType, HostTransferBuffer, HostTransferPolicy, Stream,
};

#[test]
fn retained_storage_merges_physical_aliases_and_outlives_original_handles() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let values = (0..120).map(|n| n as f32 * 0.25 - 7.0).collect::<Vec<_>>();
    let root = Array::from_slice(&values, &[2, 20, 3]);
    let view = root.try_index_device((.., 18.., ..), &stream).unwrap();
    view.evaluated().unwrap();
    let native_bytes = root.allocation_info().unwrap().unwrap().bytes() as u64;
    let mut host =
        HostTransferBuffer::new(&[2], safemlx::Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
    let host_values = [7i32, -3]
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect::<Vec<_>>();
    host.as_bytes_mut().unwrap().copy_from_slice(&host_values);
    let host = Arc::new(host.freeze());
    let host_bytes = host.capacity().unwrap() as u64;
    let weak_host = Arc::downgrade(&host);
    let mut bytes = Vec::with_capacity(4096);
    bytes.extend([1, 2]);
    let source_bytes = bytes.capacity() as u64;
    let source = MemoryWeightStore::from_safetensors([(
        "weight".into(),
        safetensors::Dtype::U8,
        vec![2],
        bytes,
    )])
    .unwrap();

    let mut inventory = RetainedStorage::default();
    inventory.include_array(&root).unwrap();
    inventory.include_array(&view).unwrap();
    inventory.include_host(Arc::clone(&host)).unwrap();
    inventory
        .include_sources(source.source_storage().unwrap())
        .unwrap();
    let mut aliases = RetainedStorage::default();
    aliases.include_array(&view).unwrap();
    aliases.include_host(host).unwrap();
    aliases
        .include_sources(source.source_storage().unwrap())
        .unwrap();
    inventory.merge(aliases).unwrap();
    assert_eq!(
        inventory.byte_bound().unwrap(),
        Some(native_bytes + host_bytes + source_bytes)
    );
    assert_eq!(inventory.arrays.len(), 1);
    assert_eq!(inventory.hosts.len(), 1);
    drop((root, view, source));
    assert_eq!(
        inventory
            .arrays
            .values()
            .next()
            .unwrap()
            .1
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        values
    );
    assert_eq!(
        weak_host.upgrade().unwrap().as_bytes().unwrap(),
        host_values
    );
    drop(inventory);
    assert!(weak_host.upgrade().is_none());
}

#[test]
fn retained_storage_keeps_unfinished_and_unpriced_owners_unknown() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2.0f32, -3.0], &[2]);
    let lazy = root.square(&stream).unwrap();
    assert_eq!(lazy.allocation_info().unwrap(), None);
    let mut inventory = RetainedStorage::default();
    inventory.include_array(&lazy).unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), None);
    assert_eq!(lazy.allocation_info().unwrap(), None);
    drop(lazy);
    assert_eq!(
        inventory.unknown_arrays[0]
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        [4.0, 9.0]
    );
    // A snapshot cannot silently become an admission after somebody evaluates.
    assert_eq!(inventory.byte_bound().unwrap(), None);
    let mut unknown_source = RetainedStorage::default();
    unknown_source.include_sources(None).unwrap();
    let mut known = RetainedStorage::default();
    known.include_array(&root).unwrap();
    known.merge(unknown_source).unwrap();
    assert_eq!(known.byte_bound().unwrap(), None);
}

#[test]
fn retained_storage_checks_capacity_conflicts_and_sum_overflow() {
    assert!(require_same_capacity(4, 8).is_err());
    let mut inventory = RetainedStorage::default();
    inventory.sources.insert(Arc::new(1u8), u64::MAX).unwrap();
    inventory.sources.insert(Arc::new(2u8), 1).unwrap();
    assert!(inventory.byte_bound().is_err());
}

#[test]
fn retained_storage_deduplicates_certified_host_array_backing_in_either_order() {
    let devices = if cfg!(feature = "metal") {
        vec![DeviceType::Cpu, DeviceType::Gpu]
    } else {
        vec![DeviceType::Cpu]
    };
    for device in devices {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut buffer = HostTransferBuffer::new(
            &[4, 8],
            safemlx::Dtype::Float32,
            HostTransferPolicy::Transfer,
        )
        .unwrap();
        let values = (0..32).map(|n| n as f32 * 0.5 - 3.0).collect::<Vec<_>>();
        buffer.as_bytes_mut().unwrap().copy_from_slice(
            &values
                .iter()
                .flat_map(|n| n.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let buffer = Arc::new(buffer.freeze());
        let host = buffer.allocation_info().unwrap();
        let array = buffer
            .copy_to_array(&stream)
            .unwrap()
            .synchronize()
            .unwrap();
        array.evaluated().unwrap();
        let native = array.allocation_info().unwrap().unwrap();
        let view = array.try_index_device((2.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        let expected = host.bytes() as u64
            + if host.identity() == native.identity() {
                0
            } else {
                native.bytes() as u64
            };
        if cfg!(feature = "metal") && device == DeviceType::Gpu {
            assert_eq!(host.identity(), native.identity());
        }
        let mut first = RetainedStorage::default();
        first.include_host(Arc::clone(&buffer)).unwrap();
        first.include_array(&view).unwrap();
        let mut second = RetainedStorage::default();
        second.include_array(&array).unwrap();
        second.include_host(Arc::clone(&buffer)).unwrap();
        assert_eq!(first.byte_bound().unwrap(), Some(expected));
        assert_eq!(second.byte_bound().unwrap(), Some(expected));
        first.merge(second).unwrap();
        assert_eq!(first.byte_bound().unwrap(), Some(expected));
        drop((buffer, array));
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &values[16..]);
        drop(view);
        assert_eq!(first.byte_bound().unwrap(), Some(expected));
    }
}
