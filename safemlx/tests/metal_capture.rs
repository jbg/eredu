#![cfg(all(feature = "metal", target_vendor = "apple"))]

/// Native export, nesting and drop cleanup need Metal capture enabled at launch.
#[test]
#[ignore = "requires MTL_CAPTURE_ENABLED=1 and native Metal access"]
fn capture_exports_completed_work_and_drop_allows_retry() {
    use safemlx::{metal::Capture, Array, Device, DeviceType, Stream};
    let directory = tempfile::tempdir().unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let input = Array::from_slice(&[1f32, 2., 3.], &[3]);
    safemlx::transforms::eval([&input]).unwrap();
    stream.synchronize().unwrap();
    let first = directory.path().join("first.gputrace");
    let capture = Capture::begin(&first).unwrap();
    assert!(Capture::begin(directory.path().join("nested.gputrace")).is_err());
    let output = input.add(&input, &stream).unwrap();
    let timing = safemlx::transforms::async_eval_timed([&output], &stream).unwrap();
    timing.elapsed().unwrap();
    let output = output.into_evaluated().unwrap();
    assert_eq!(output.as_slice::<f32>(), &[2., 4., 6.]);
    stream.synchronize().unwrap();
    capture.finish().unwrap();
    assert!(first.exists());
    assert!(Capture::begin(&first).is_err());
    drop(Capture::begin(directory.path().join("drop.gputrace")).unwrap());
    Capture::begin(directory.path().join("retry.gputrace"))
        .unwrap()
        .finish()
        .unwrap();
}
