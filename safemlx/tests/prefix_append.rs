use safemlx::{ops, Array, Device, DeviceType, PrefixAppendBuffer, Stream};

fn values(array: Array) -> Vec<f32> {
    array.into_evaluated().unwrap().try_to_vec().unwrap()
}
fn check(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    let other = Stream::new_with_device(&Device::new(device, 0));
    let initial = Array::from_slice(&[1f32, 2., 3., 4.], &[2, 1, 1, 2]);
    let buffer = PrefixAppendBuffer::new(&initial, 4, -2, &stream).unwrap();
    let prefix = buffer.prefix(&stream).unwrap().into_evaluated().unwrap();
    assert!(prefix.try_as_slice::<f32>().is_err());
    assert_eq!(prefix.try_to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
    let snapshot = prefix.deep_clone().unwrap();
    let update = Array::from_slice(&[5f32, 6., 7., 8.], &[2, 1, 1, 2]);
    let candidate = buffer.appended(&update, 4, &stream).unwrap();
    let appended = candidate.prefix(&stream).unwrap().into_evaluated().unwrap();
    let storage = safemlx::inspect_storage(&[prefix.as_array(), appended.as_array()]).unwrap();
    assert!(storage.values[0].is_some());
    assert_eq!(
        storage.values[0], storage.values[1],
        "a reserved suffix reuses the committed backing rather than copying it"
    );
    // A dropped candidate still consumes its reservation. A delayed execution
    // from another descendant cannot overwrite the selected branch.
    let abandoned = candidate.appended(&update, 4, &other).unwrap();
    let next_update = Array::from_slice(&[9f32, 10., 11., 12.], &[2, 1, 1, 2]);
    assert!(!candidate.can_append(1));
    let committed = candidate.appended(&next_update, 4, &stream).unwrap();
    assert_eq!(
        values(committed.prefix(&stream).unwrap()),
        [1., 2., 5., 6., 9., 10., 3., 4., 7., 8., 11., 12.]
    );
    assert_eq!(
        values(abandoned.prefix(&other).unwrap()),
        [1., 2., 5., 6., 5., 6., 3., 4., 7., 8., 7., 8.]
    );
    assert_eq!(prefix.try_to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
    assert_eq!(snapshot.try_to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
    let truncated = committed.truncated(1).unwrap();
    let restored = truncated.appended(&next_update, 4, &other).unwrap();
    assert_eq!(
        values(restored.prefix(&other).unwrap()),
        [1., 2., 9., 10., 3., 4., 11., 12.]
    );
    let grown = committed
        .appended(&update, 4, &stream)
        .unwrap()
        .appended(&next_update, 8, &stream)
        .unwrap();
    assert_eq!(grown.capacity(), 8);
    assert_eq!(
        values(grown.prefix(&stream).unwrap()),
        [1., 2., 5., 6., 9., 10., 5., 6., 9., 10., 3., 4., 7., 8., 11., 12., 7., 8., 11., 12.]
    );
    assert!(ops::as_strided(
        prefix.as_array(),
        Some(&[9][..]),
        Some(&[1][..]),
        Some(0),
        &stream
    )
    .is_err());
    let contiguous = Array::from_slice(&[1f32, 2., 3.], &[3]);
    let reverse = ops::as_strided(
        &contiguous,
        Some(&[3][..]),
        Some(&[-1][..]),
        Some(2),
        &stream,
    )
    .unwrap();
    assert_eq!(values(reverse), [3., 2., 1.]);
    let small = Array::from_slice(&[13f32, 14.], &[1, 1, 1, 2]);
    let single = PrefixAppendBuffer::new(&small, 4, -2, &stream).unwrap();
    let exposed = single.prefix(&stream).unwrap().into_evaluated().unwrap();
    let borrowed = exposed.try_as_slice::<f32>().unwrap();
    let next = single.appended(&small, 4, &other).unwrap();
    let output = next.prefix(&other).unwrap();
    let completion = safemlx::transforms::async_eval_with_event([&output]).unwrap();
    drop(next);
    drop(single);
    completion.synchronize().unwrap();
    assert_eq!(borrowed, [13., 14.]);
    assert_eq!(values(output), [13., 14., 13., 14.]);
    stream.synchronize().unwrap();
    other.synchronize().unwrap();
}
#[test]
fn prefixes_preserve_rollback_and_owned_host_reads_cpu() {
    check(DeviceType::Cpu);
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn prefixes_preserve_rollback_and_owned_host_reads_metal() {
    check(DeviceType::Gpu);
}

#[test]
fn prefix_mutation_rejects_compiled_replay() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut compiled = safemlx::transforms::compile::compile_unary_with_stream(
        |input, stream| {
            PrefixAppendBuffer::new(input, 4, 0, stream)?
                .appended(input, 4, stream)?
                .prefix(stream)
        },
        false,
    );
    let input = Array::from_slice(&[1f32], &[1]);
    assert!(compiled.call(&input, &stream).is_err());
}
