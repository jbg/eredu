//! Shared cold predicates for native convolution execution and its byte quote.

pub(crate) fn uses_metal_winograd_2d(
    input: &[i32],
    weight: &[i32],
    stride: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
) -> bool {
    input.len() == 4
        && weight.len() == 4
        && input.iter().chain(weight).all(|&v| v > 0)
        && groups == 1
        && stride == (1, 1)
        && dilation == (1, 1)
        && weight[1..3] == [3, 3]
        && input[3] == weight[3]
        && input[3] % 32 == 0
        && weight[0] % 32 == 0
        && input[0] as u128 * input[1] as u128 * input[2] as u128 >= 4096
        && i64::from(input[3]) + i64::from(weight[0]) >= 256
}

pub(crate) fn transpose_1d_needs_output_crop(
    weight: &[i32],
    stride: i32,
    padding: i32,
    dilation: i32,
) -> bool {
    weight.len() == 3
        && weight[1] > 0
        && stride > 1
        && dilation > 0
        && padding >= 0
        && i64::from(padding) > i64::from(dilation) * (i64::from(weight[1]) - 1)
}

pub(crate) mod original;
