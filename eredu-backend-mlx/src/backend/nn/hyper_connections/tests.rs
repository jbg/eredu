use super::split_sinkhorn;
use crate::backend::ExecutionContext;
use safemlx::{Array, Device, DeviceType};

#[test]
#[ignore = "requires MLX runtime execution"]
fn preserves_arbitrary_leading_dimensions_and_normalizes_columns() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mixes = Array::from_slice(&[0.0f32; 48], &[2, 3, 8]);
    let scale = Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]);
    let base = Array::from_slice(&[0.0f32; 8], &[8]);
    let split = split_sinkhorn(&mixes, &scale, &base, 2, 20, 1e-6, stream).unwrap();

    assert_eq!(split.pre.shape(), [2, 3, 2]);
    assert_eq!(split.post.shape(), [2, 3, 2]);
    assert_eq!(split.combination.shape(), [2, 3, 2, 2]);
    let columns = split.combination.sum_axis(-2, false, stream).unwrap();
    assert!(columns
        .all_close(
            Array::ones::<f32>(&[2, 3, 2], stream).unwrap(),
            1e-4,
            1e-4,
            None,
            stream,
        )
        .unwrap()
        .item::<bool>(stream));
}
