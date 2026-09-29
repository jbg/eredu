//! This contains the tests for some of the exported macros.
//!
//! This is mainly a sanity check to ensure that the exported macros are working as expected.

use safemlx::{array, Dtype};

mod common;

#[test]
fn test_ops_convolution_conv1d() {
    let input = array!(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        shape = [1, 5, 2]
    );
    let weight = array!(
        [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0],
        shape = [2, 3, 2]
    );

    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let result = safemlx::conv1d!(
        &input,
        &weight,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        stream = &stream
    )
    .unwrap();

    let expected = array!([12.0, 8.0, 17.0, 13.0, 22.0, 18.0], shape = [1, 3, 2]);
    assert!(common::eval_equal_values(&result, &expected));
}

#[test]
fn test_ops_factory_arange() {
    // Without specifying start and step
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let array = safemlx::arange!(stop = 50, stream = &stream).unwrap();
    assert_eq!(array.shape(), &[50]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: Vec<f32> = common::eval_vec(&array);
    let expected: Vec<f32> = (0..50).map(|x| x as f32).collect();
    assert_eq!(data, expected);

    // With specifying start and step
    let array = safemlx::arange!(start = 1.0, stop = 50.0, step = 2.0, stream = &stream).unwrap();
    assert_eq!(array.shape(), &[25]);
    assert_eq!(array.dtype(), Dtype::Float32);

    let data: Vec<f32> = common::eval_vec(&array);
    let expected: Vec<f32> = (1..50).step_by(2).map(|x| x as f32).collect();
    assert_eq!(data, expected);


}

// Test functions defined in `safemlx::fft` module.

// Test functions defined in `safemlx::linalg` module.

// Test functions defined in `safemlx::random` module.

// Test functions defined in `safemlx::fast` module.
