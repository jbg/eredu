use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use safemlx_internal_macros::generate_macro;

/// General convolution over an input with several channels returning an error if the inputs are invalid.
///
/// - Only 1d and 2d convolutions are supported at the moment
/// - the default `groups: 1` is currently supported
///
/// # Params
///
/// - array: Input array of shape `&[N, ..., C_in]`
/// - weight: Weight array of shape `&[C_out, ..., C_in]`
/// - strides: The kernel strides. All dimensions get the same stride if only one number is specified.
/// - padding: The input padding. All dimensions get the same padding if only one number is specified.
/// - kernel_dilation: The kernel dilation. All dimensions get the same dilation if only one number is specified.
/// - input_dilation: The input dilation. All dimensions get the same dilation if only one number is specified.
/// - groups: Input feature groups
/// - flip: Flip the order in which the spatial dimensions of the weights are processed.
///   Performs the cross-correlation operator when `flip` is `false` and the convolution
///   operator otherwise.
#[generate_macro]
#[allow(clippy::too_many_arguments)]
pub fn conv_general<'a>(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] strides: impl IntoOption<&'a [i32]>,
    #[optional] padding: impl IntoOption<&'a [i32]>,
    #[optional] kernel_dilation: impl IntoOption<&'a [i32]>,
    #[optional] input_dilation: impl IntoOption<&'a [i32]>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] flip: impl Into<Option<bool>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let strides = strides.into_option().unwrap_or(&[1]);
    let padding = padding.into_option().unwrap_or(&[0]);
    let kernel_dilation = kernel_dilation.into_option().unwrap_or(&[1]);
    let input_dilation = input_dilation.into_option().unwrap_or(&[1]);
    let groups = groups.into().unwrap_or(1);
    let flip = flip.into().unwrap_or(false);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv_general(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            strides.as_ptr(),
            strides.len(),
            padding.as_ptr(),
            padding.len(),
            padding.as_ptr(),
            padding.len(),
            kernel_dilation.as_ptr(),
            kernel_dilation.len(),
            input_dilation.as_ptr(),
            input_dilation.len(),
            groups,
            flip,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 1D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// # Params
///
/// - array: input array of shape `&[N, H, C_in]`
/// - weight: weight array of shape `&[C_out, H, C_in]`
/// - stride: kernel stride. Default to 1 if not specified.
/// - padding: input padding. Default to 0 if not specified.
/// - dilation: kernel dilation. Default to 1 if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
#[generate_macro]
pub fn conv1d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv1d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride,
            padding,
            dilation,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 2D convolution over an input with several channels returning an error if the inputs are invalid.
///
/// Only the default `groups=1` is currently supported.
///
/// # Params
///
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1) if not specified.
/// - padding: input padding. Default to (0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
#[generate_macro]
pub fn conv2d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1));
    let padding = padding.into().unwrap_or((0, 0));
    let dilation = dilation.into().unwrap_or((1, 1));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv2d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 3D convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
#[generate_macro]
pub fn conv3d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1, 1));
    let padding = padding.into().unwrap_or((0, 0, 0));
    let dilation = dilation.into().unwrap_or((1, 1, 1));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv3d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            stride.2,
            padding.0,
            padding.1,
            padding.2,
            dilation.0,
            dilation.1,
            dilation.2,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 1D transposed convolution over an input with several channels.
///
/// # Params
///
/// - array: input array of shape `[N, H, C_in]`
/// - weight: weight array of shape `[C_out, H, C_in]`
/// - stride: kernel stride. Default to 1 if not specified.
/// - padding: input padding. Default to 0 if not specified.
/// - dilation: kernel dilation. Default to 1 if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn conv_transpose1d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<i32>>,
    #[optional] padding: impl Into<Option<i32>>,
    #[optional] dilation: impl Into<Option<i32>>,
    #[optional] output_padding: impl Into<Option<i32>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or(1);
    let padding = padding.into().unwrap_or(0);
    let dilation = dilation.into().unwrap_or(1);
    let output_padding = output_padding.into().unwrap_or(0);
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv_transpose1d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 2D transposed convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// # Params
/// - array: input array of shape `[N, H, W, C_in]`
/// - weight: weight array of shape `[C_out, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1) if not specified.
/// - padding: input padding. Default to (0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn conv_transpose2d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32)>>,
    #[optional] output_padding: impl Into<Option<(i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1));
    let padding = padding.into().unwrap_or((0, 0));
    let dilation = dilation.into().unwrap_or((1, 1));
    let output_padding = output_padding.into().unwrap_or((0, 0));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv_transpose2d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            output_padding.0,
            output_padding.1,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

/// 3D transposed convolution over an input with several channels.
///
/// Only the default `groups=1` is currently supported.
///
/// The numeric parameters may be given as single values:
///
/// # Params
/// - array: input array of shape `[N, D, H, W, C_in]`
/// - weight: weight array of shape `[C_out, D, H, W, C_in]`
/// - stride: kernel stride. Default to (1, 1, 1) if not specified.
/// - padding: input padding. Default to (0, 0, 0) if not specified.
/// - dilation: kernel dilation. Default to (1, 1, 1) if not specified.
/// - groups: input feature groups. Default to 1 if not specified.
/// - stream: stream or device to evaluate on.
#[allow(clippy::too_many_arguments)]
#[generate_macro]
pub fn conv_transpose3d(
    array: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    #[optional] stride: impl Into<Option<(i32, i32, i32)>>,
    #[optional] padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] dilation: impl Into<Option<(i32, i32, i32)>>,
    #[optional] output_padding: impl Into<Option<(i32, i32, i32)>>,
    #[optional] groups: impl Into<Option<i32>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stride = stride.into().unwrap_or((1, 1, 1));
    let padding = padding.into().unwrap_or((0, 0, 0));
    let dilation = dilation.into().unwrap_or((1, 1, 1));
    let output_padding = output_padding.into().unwrap_or((0, 0, 0));
    let groups = groups.into().unwrap_or(1);

    Array::try_from_op(|res| unsafe {
        safemlx_sys::mlx_conv_transpose3d(
            res,
            array.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            stride.0,
            stride.1,
            stride.2,
            padding.0,
            padding.1,
            padding.2,
            dilation.0,
            dilation.1,
            dilation.2,
            output_padding.0,
            output_padding.1,
            output_padding.2,
            groups,
            stream.as_ref().as_ptr(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_conv1d_complex_device() {
        let stream = crate::test_stream();
        // Define a 1D input with two channels
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let input_array = Array::from_slice(&input_data, &[1, 5, 2]);

        // Define a 1D kernel with two input channels and two output channels
        let weight_data = [0.5, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0];
        let weight_array = Array::from_slice(&weight_data, &[2, 3, 2]);

        let result = conv1d(
            &input_array,
            &weight_array,
            Some(1), // stride
            Some(0), // padding
            Some(1), // dilation
            Some(1), // groups
            stream,
        )
        .unwrap();

        let expected_output = [12.0, 8.0, 17.0, 13.0, 22.0, 18.0];
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(crate::array::eval_vec::<f32>(&result), &expected_output);
    }

    #[test]
    fn test_conv_transpose1d() {
        let stream = crate::test_stream();
        // Single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0], &[1, 3, 1]);
        // Single input/output channel kernel
        let weights = Array::from_slice(&[1.0, 0.5], &[1, 2, 1]);

        let result = conv_transpose1d(
            &input,
            &weights,
            Some(1), // stride
            Some(0), // padding
            Some(1), // dilation
            None,    // output padding
            Some(1), // groups
            stream,
        )
        .unwrap();

        let expected = [1.0, 2.5, 4.0, 1.5];
        assert_eq!(result.shape(), &[1, 4, 1]);
        assert_eq!(crate::array::eval_vec::<f32>(&result), &expected);
    }

    #[test]
    fn test_grouped_conv_transpose1d() {
        let stream = crate::test_stream();
        // Two independent depthwise channels. Values are interleaved by timestep.
        let input = Array::from_slice(&[1.0, 10.0, 2.0, 20.0], &[1, 2, 2]);
        let weights = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);

        let result = conv_transpose1d(
            &input,
            &weights,
            Some(1),
            Some(0),
            Some(1),
            None,
            Some(2),
            stream,
        )
        .unwrap();

        let expected = [1.0, 30.0, 4.0, 100.0, 4.0, 80.0];
        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(crate::array::eval_vec::<f32>(&result), &expected);
    }

    #[test]
    fn test_conv2d() {
        let stream = crate::test_stream();
        // Define a 2x2 input with one channel (grayscale image or similar)
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // Perform the convolution with no padding and stride of 1
        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
            stream,
        )
        .unwrap();

        // Expected result is the convolution of a 2x2 filter over a 2x2 input with valid padding, resulting in a single output value
        let expected_output = 1.0 * 1.0 + 2.0 * 0.0 + 3.0 * 0.0 + 4.0 * 1.0; // = 1*1 + 4*1 = 5
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[expected_output]);
    }

    #[test]
    fn test_conv_transpose2d() {
        let stream = crate::test_stream();
        // 2x2 single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
        // 2x2 single channel kernel (identity-like)
        let weights = Array::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2, 1]);

        let result = conv_transpose2d(
            &input,
            &weights,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            None,         // output padding
            Some(1),      // groups
            stream,
        )
        .unwrap();

        let expected = [1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0];
        assert_eq!(result.shape(), &[1, 3, 3, 1]);
        assert_eq!(crate::array::eval_vec::<f32>(&result), &expected);
    }

    #[test]
    fn test_conv3d() {
        let stream = crate::test_stream();
        // Define a 2x2x2 input with one channel
        let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input_shape = [1, 2, 2, 2, 1]; // [N, D, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        // Define a 2x2x2 kernel with one input channel and one output channel
        let weight_data = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let weight_shape = [1, 2, 2, 2, 1]; // [C_out, D_k, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        // Perform the convolution with no padding and stride of 1
        let result = conv3d(
            &input_array,
            &weight_array,
            Some((1, 1, 1)), // stride
            Some((0, 0, 0)), // padding
            Some((1, 1, 1)), // dilation
            Some(1),         // groups
            stream,
        )
        .unwrap();

        // Expected result is the convolution of a 2x2x2 filter over a 2x2x2 input with valid padding, resulting in a single output value
        let expected_output = 1.0 * 1.0
            + 2.0 * 0.0
            + 3.0 * 0.0
            + 4.0 * 1.0
            + 5.0 * 0.0
            + 6.0 * 1.0
            + 7.0 * 1.0
            + 8.0 * 0.0; // = 1*1 + 4*1 + 6*1 + 7*1 = 18
        assert_eq!(crate::array::eval_vec::<f32>(&result), &[expected_output]);
    }

    #[test]
    fn test_conv_transpose3d() {
        let stream = crate::test_stream();
        // 2x2x2 single channel input
        let input = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 2, 2, 2, 1]);
        // 2x2x2 single channel kernel
        let weights =
            Array::from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[1, 2, 2, 2, 1]);

        let result = conv_transpose3d(
            &input,
            &weights,
            Some((1, 1, 1)), // stride
            Some((0, 0, 0)), // padding
            Some((1, 1, 1)), // dilation
            None,            // output padding
            Some(1),         // groups
            stream,
        )
        .unwrap();

        assert_eq!(result.shape(), &[1, 3, 3, 3, 1]);
    }

    #[test]
    fn test_conv_wrong_dimensions() {
        let stream = crate::test_stream();
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2]; // [C_out, H_k, W_k]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
            stream,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_invalid_group_size() {
        let stream = crate::test_stream();
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1.0, 0.0, 0.0, 1.0];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(2),      // groups
            stream,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_conv_non_float() {
        let stream = crate::test_stream();
        let input_data = [1, 2, 3, 4];
        let input_shape = [1, 2, 2, 1]; // [N, H, W, C]
        let input_array = Array::from_slice(&input_data, &input_shape);

        let weight_data = [1, 0, 0, 1];
        let weight_shape = [1, 2, 2, 1]; // [C_out, H_k, W_k, C_in]
        let weight_array = Array::from_slice(&weight_data, &weight_shape);

        let result = conv2d(
            &input_array,
            &weight_array,
            Some((1, 1)), // stride
            Some((0, 0)), // padding
            Some((1, 1)), // dilation
            Some(1),      // groups
            stream,
        );

        assert!(result.is_err());
    }
}

/// Actual compiled implicit/separable convolution construction and worker facts.
/// This inspects source geometry only; it creates no tensor, context or authority.
#[derive(Clone, Copy, Debug)]
pub struct OriginalConvolutionLayout {
    native: safemlx_sys::mlx_original_convolution_layout,
    rank: usize,
}
impl OriginalConvolutionLayout {
    /// Inspect a nontransposed 1D/2D convolution. Other selected algorithms
    /// return None until their complete native construction/worker receipt exists.
    pub fn inspect(
        input: &[i32],
        weight: &[i32],
        stride: &[i32],
        padding: &[i32],
        dilation: &[i32],
        groups: i32,
    ) -> Option<Self> {
        let dims = stride.len();
        if !(1..=2).contains(&dims)
            || input.len() != dims + 2
            || weight.len() != dims + 2
            || padding.len() != dims
            || dilation.len() != dims
        {
            return None;
        }
        let mut source = safemlx_sys::mlx_original_convolution_source {
            spatial_dimensions: dims,
            input: [
                input[0],
                input[1],
                if dims == 2 { input[2] } else { 1 },
                input[dims + 1],
            ],
            weight: [
                weight[0],
                weight[1],
                if dims == 2 { weight[2] } else { 1 },
                weight[dims + 1],
            ],
            stride: [1; 2],
            padding: [0; 2],
            dilation: [1; 2],
            groups,
        };
        source.stride[..dims].copy_from_slice(stride);
        source.padding[..dims].copy_from_slice(padding);
        source.dilation[..dims].copy_from_slice(dilation);
        let mut native = safemlx_sys::mlx_original_convolution_layout::default();
        // SAFETY: both structures live for the call; native reads fixed source
        // fields and writes this output only. No TLS/native object is consulted.
        if !unsafe { safemlx_sys::mlx_original_convolution_inspect(&mut native, &source) } {
            return None;
        }
        if dims == 1 {
            native.output[2] = native.output[3];
            native.output[3] = 1;
        }
        Some(Self {
            native,
            rank: dims + 2,
        })
    }
    /// Exact output geometry established by the native constructor formula.
    pub fn output_shape(&self) -> &[i32] {
        &self.native.output[..self.rank]
    }
    /// Constructor descriptor population, including its possible dtype casts.
    pub fn primitives(self) -> usize {
        self.native.primitives
    }
    /// Constructor input-edge population.
    pub fn edges(self) -> usize {
        self.native.edges
    }
    /// Output/cast births and the two possible row-contiguous input copies.
    pub fn backing_births(self) -> usize {
        self.native.backing_births
    }
    /// Named native/safe constructor and query controls, excluding the shared
    /// per-kernel selector population already paid by the operation recipe.
    pub fn control_bytes(self) -> Option<usize> {
        use crate::OriginalScopeObserver;
        use crate::utils::{guard::MaybeUninitArray, runtime_lock::RuntimeLockGuard};
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<safemlx_sys::mlx_original_convolution_source>(),
            size_of::<safemlx_sys::mlx_original_convolution_layout>(),
            size_of::<[&[i32]; 5]>(),
            size_of::<[usize; 2]>(),
            size_of::<i32>(),
            size_of::<[&Array; 2]>(),
            size_of::<&Stream>(),
            size_of::<[Option<(i32, i32)>; 3]>(),
            size_of::<Option<i32>>(),
            size_of::<Array>(),
            size_of::<MaybeUninitArray>(),
            size_of::<Result<Array>>(),
            size_of::<RuntimeLockGuard>(),
            size_of::<Option<RuntimeLockGuard>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<Result<Option<OriginalScopeObserver>>>(),
        ];
        frames.into_iter().try_fold(
            self.native
                .named_control_bytes
                .checked_add(size_of_val(&frames))?,
            usize::checked_add,
        )
    }
}
