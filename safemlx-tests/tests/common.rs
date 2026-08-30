#![allow(dead_code)]

use safemlx::{Array, ArrayElement, Stream};

pub fn test_stream() -> &'static Stream {
    Box::leak(Box::new(Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Cpu,
        0,
    ))))
}

pub fn eval_vec<T>(array: &Array) -> Vec<T>
where
    T: ArrayElement + Clone,
{
    array.evaluated().unwrap().as_slice::<T>().to_vec()
}

pub fn eval_equal_values(lhs: &Array, rhs: &Array) -> bool {
    let lhs = lhs.evaluated().unwrap();
    let rhs = rhs.evaluated().unwrap();
    lhs.equal_values(&rhs)
}
