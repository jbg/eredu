use safemlx::{
    ops::indexing::IndexOp, transforms::async_eval_with_event, Array, Device, DeviceType,
    EventBackend, Stream,
};

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

#[test]
fn empty_and_available_outputs_are_complete_and_identity_free() {
    let empty = async_eval_with_event(std::iter::empty()).unwrap();
    assert!(empty.is_complete().unwrap());
    assert_eq!(empty.backend().unwrap(), EventBackend::None);
    assert!(empty.device().unwrap().is_none());
    empty.synchronize().unwrap();
    empty.synchronize().unwrap();

    let available = Array::from_slice(&[1.0f32, 2.0], &[2]);
    let ready = async_eval_with_event([&available]).unwrap();
    assert!(ready.is_complete().unwrap());
    assert_eq!(ready.backend().unwrap(), EventBackend::None);
    assert!(ready.device().unwrap().is_none());
}

#[test]
fn cpu_event_is_initially_incomplete_then_monotonically_complete() {
    let producer = cpu_stream();
    let lhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
    let rhs = Array::ones::<f32>(&[1024, 1024], &producer).unwrap();
    let output = lhs.matmul(&rhs, &producer).unwrap();

    let event = async_eval_with_event([&output]).unwrap();
    assert_eq!(event.backend().unwrap(), EventBackend::Cpu);
    assert_eq!(
        event.device().unwrap().unwrap(),
        Device::new(DeviceType::Cpu, 0)
    );
    assert!(
        !event.is_complete().unwrap(),
        "the deliberately large CPU producer unexpectedly completed before its first query"
    );

    event.synchronize().unwrap();
    assert!(event.is_complete().unwrap());
    assert!(event.is_complete().unwrap());
    event.synchronize().unwrap();
}

#[test]
fn cpu_cross_stream_wait_multiple_consumers_and_raii_drop() {
    let producer = cpu_stream();
    let consumer_a = cpu_stream();
    let consumer_b = cpu_stream();
    assert_ne!(
        producer.get_index().unwrap(),
        consumer_a.get_index().unwrap()
    );
    assert_ne!(
        consumer_a.get_index().unwrap(),
        consumer_b.get_index().unwrap()
    );

    let lhs = Array::ones::<f32>(&[256, 256], &producer).unwrap();
    let rhs = Array::ones::<f32>(&[256, 256], &producer).unwrap();
    let produced = lhs.matmul(&rhs, &producer).unwrap();
    let producer_event = async_eval_with_event([&produced]).unwrap();

    consumer_a.wait_event(&producer_event).unwrap();
    producer_event.wait_on(&consumer_b).unwrap();

    let consumed_a = produced.add(Array::from(1.0f32), &consumer_a).unwrap();
    let consumed_b = produced.add(Array::from(2.0f32), &consumer_b).unwrap();
    let completion_a = async_eval_with_event([&consumed_a]).unwrap();
    let completion_b = async_eval_with_event([&consumed_b]).unwrap();

    // Both queued waits retain the native event after its public Rust handle
    // is destroyed.
    drop(producer_event);

    completion_a.synchronize().unwrap();
    completion_b.synchronize().unwrap();
    let value_a = consumed_a
        .index_device((0, 0), &consumer_a)
        .item::<f32>(&consumer_a);
    let value_b = consumed_b
        .index_device((0, 0), &consumer_b)
        .item::<f32>(&consumer_b);
    assert_eq!(value_a, 257.0);
    assert_eq!(value_b, 258.0);
}

#[cfg(feature = "metal")]
#[test]
fn incompatible_cpu_to_metal_wait_has_exact_diagnostic() {
    let cpu = cpu_stream();
    let gpu = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let output = Array::ones::<f32>(&[64, 64], &cpu)
        .unwrap()
        .square(&cpu)
        .unwrap();
    let event = async_eval_with_event([&output]).unwrap();

    let error = gpu.wait_event(&event).unwrap_err();
    let diagnostic = error.what().split(" at ").next().unwrap();
    assert_eq!(
        diagnostic,
        "[Completion::wait] Incompatible producer and consumer devices: completion is for \
Device(cpu, 0), but the consumer stream is on Device(gpu, 0)."
    );
    event.synchronize().unwrap();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "explicit Metal event handoff test; run with --ignored on a Metal host"]
fn metal_two_stream_handoff_is_gpu_ordered_without_host_synchronization() {
    let device = Device::new(DeviceType::Gpu, 0);
    let producer = Stream::new_with_device(&device);
    let consumer = Stream::new_with_device(&device);
    assert_ne!(producer.get_index().unwrap(), consumer.get_index().unwrap());

    let lhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let rhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let produced = lhs.matmul(&rhs, &producer).unwrap();
    let event = async_eval_with_event([&produced]).unwrap();
    assert_eq!(event.backend().unwrap(), EventBackend::Metal);
    assert!(!event.is_complete().unwrap());

    // If this were implemented as host synchronization, the call could not
    // return while the producer event remains incomplete.
    consumer.wait_event(&event).unwrap();
    assert!(!event.is_complete().unwrap());

    let consumed = produced.square(&consumer).unwrap();
    let consumed_event = async_eval_with_event([&consumed]).unwrap();
    drop(event);
    consumed_event.synchronize().unwrap();
    assert!(consumed_event.is_complete().unwrap());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit CUDA event handoff test; requires a CUDA-capable host"]
fn cuda_two_stream_handoff_uses_cuda_stream_wait() {
    let device = Device::new(DeviceType::Gpu, 0);
    let producer = Stream::new_with_device(&device);
    let consumer = Stream::new_with_device(&device);
    let lhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let rhs = Array::ones::<f32>(&[4096, 4096], &producer).unwrap();
    let produced = lhs.matmul(&rhs, &producer).unwrap();
    let event = async_eval_with_event([&produced]).unwrap();
    assert_eq!(event.backend().unwrap(), EventBackend::Cuda);
    assert!(!event.is_complete().unwrap());
    consumer.wait_event(&event).unwrap();
    assert!(!event.is_complete().unwrap());
    let consumed = produced.square(&consumer).unwrap();
    async_eval_with_event([&consumed])
        .unwrap()
        .synchronize()
        .unwrap();
}
