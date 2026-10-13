//! Native gated arithmetic retaining the ordinary MLX FP32 operation boundaries.
//! Exact `dispatch_threads` grids include partial threadgroups, so elementwise
//! kernels do not specialize on the array length or need a padded-grid guard.
use safemlx::{error::Exception, Array, Stream};

#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::{
    fast::{CustomKernelConfig, MetalKernel},
    DeviceType, Dtype,
};

/// Fuses the pointwise sides of grouped RMS without changing its sum reduction.
pub(crate) fn grouped_rms(
    input: &Array,
    gate: &Array,
    weight: &Array,
    groups: i32,
    epsilon: f32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use std::cell::RefCell;
        if input.ndim() == 0
            || input.size() == 0
            || groups <= 0
            || input.dim(-1) % groups != 0
            || input.shape() != gate.shape()
            || weight.shape() != [input.dim(-1)]
            || [input, gate, weight]
                .iter()
                .any(|x| x.dtype() != Dtype::Float32)
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let Ok(size) = i32::try_from(input.size()) else {
            return Ok(None);
        };
        let width = input.dim(-1) / groups;
        thread_local! {
            static PRE: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
            static POST: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }
        let mut outputs = PRE.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_gated_rms_inputs",
                    ["input", "gate"],
                    ["gated", "squared"],
                    r#"
                    uint i=thread_position_in_grid.x;
                    float x=gate[i];
                    // MLX's ordinary sigmoid, distinct from stable_exp_f32 SiLU.
                    float y=1.0f/(1.0f+metal::precise::exp(metal::abs(x)));
                    float probability=x<0 ? y : 1.0f-y;
                    float activated=x*probability;
                    float value=input[i]*activated;
                    gated[i]=value; squared[i]=value*value;
                    "#,
                    "#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("gated RMS input kernel initialized")
                .apply_device(
                    [input, gate],
                    &CustomKernelConfig::new()
                        .with_grid([size, 1, 1])
                        .with_thread_group([256, 1, 1])
                        .with_output_arg(input.shape(), Dtype::Float32)
                        .with_output_arg(input.shape(), Dtype::Float32),
                    stream,
                )
        })?;
        let squared = outputs.pop().expect("squared output");
        let gated = outputs.pop().expect("gated output");
        // Retain native sum selection, including its rounding and shape-dependent
        // reduction order. The retained native mean divides once by its element count.
        let sums = squared
            .reshape(&[-1, groups, width], stream)?
            .sum_axis(-1, true, stream)?;
        let scalars = Array::from_slice(&[epsilon, width as f32], &[2]);
        let output = POST.with(|cell| -> Result<Array, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_gated_rms_output",
                    ["gated", "sums", "weight", "scalars"],
                    ["output"],
                    r#"
                    uint i=thread_position_in_grid.x;
                    uint row=i/WIDTH;
                    float variance=sums[row]/scalars[1];
                    float inverse=metal::precise::rsqrt(variance+scalars[0]);
                    float normalized=gated[i]*inverse;
                    output[i]=normalized*weight[i%(GROUPS*WIDTH)];
                    "#,
                    "#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("gated RMS output kernel initialized")
                .apply_one_device(
                    [&gated, &sums, weight, &scalars],
                    &CustomKernelConfig::new()
                        .with_template_arg_int("WIDTH", width)
                        .with_template_arg_int("GROUPS", groups)
                        .with_grid([size, 1, 1])
                        .with_thread_group([256, 1, 1])
                        .with_output_arg(input.shape(), Dtype::Float32),
                    stream,
                )
        })?;
        Ok(Some(output))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, gate, weight, groups, epsilon, stream);
        Ok(None)
    }
}

/// Tanh-GELU followed by an independent product, with FP32 intermediates.
pub(crate) fn gelu_product(
    gate: &Array,
    up: &Array,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use std::cell::RefCell;
        if gate.ndim() == 0
            || gate.size() == 0
            || gate.shape() != up.shape()
            || gate.dtype() != Dtype::Float32
            || up.dtype() != Dtype::Float32
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let Ok(size) = i32::try_from(gate.size()) else {
            return Ok(None);
        };
        // Retain the ordinary GELU multiplication order and its host-rounded
        // coefficient, including the reference tanh saturation boundary.
        let scalars = Array::from_slice(
            &[
                0.044715_f32,
                0.5,
                1.0,
                (2.0_f64 / std::f64::consts::PI).sqrt() as f32,
                8.664_34,
            ],
            &[5],
        );
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let output = KERNEL.with(|cell| -> Result<Array, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "f32_gelu_product",
                    ["gate", "up", "scalars"],
                    ["output"],
                    r#"
                    uint i=thread_position_in_grid.x;
                    float x=gate[i];
                    float square=x*x;
                    float cube=square*x;
                    float correction=cube*scalars[0];
                    float inner=x+correction;
                    float argument=inner*scalars[3];
                    float hyperbolic=argument>scalars[4] ? scalars[2] :
                        (argument<-scalars[4] ? -scalars[2] : metal::precise::tanh(argument));
                    float curved=hyperbolic+scalars[2];
                    float half_product=scalars[1]*x;
                    float activated=half_product*curved;
                    output[i]=activated*up[i];
                    "#,
                    "#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)\n",
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("gated GELU kernel initialized")
                .apply_one_device(
                    [gate, up, &scalars],
                    &CustomKernelConfig::new()
                        .with_grid([size, 1, 1])
                        .with_thread_group([256, 1, 1])
                        .with_output_arg(gate.shape(), Dtype::Float32),
                    stream,
                )
        })?;
        Ok(Some(output))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (gate, up, stream);
        Ok(None)
    }
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{ops::indexing::TryIndexOp, Device};

    fn same_bits(actual: Array, expected: Array) {
        assert_eq!(actual.shape(), expected.shape());
        let actual = actual.into_evaluated().unwrap();
        let expected = expected.into_evaluated().unwrap();
        for (i, (a, b)) in actual
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .enumerate()
        {
            assert!(
                a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                "element {i}: {a} ({:x}) vs {b} ({:x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    #[test]
    fn grouped_rms_preserves_native_rounding_groups_strides_and_tails() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        for width in [17, 72, 257, 1280, 10240] {
            let groups = 3;
            let dimensions = width * groups;
            let values = (0..4 * dimensions)
                .map(|i| ((i * 37 % 103) as f32 - 51.) / 13.)
                .collect::<Vec<_>>();
            let gates = values.iter().rev().copied().collect::<Vec<_>>();
            let weights = (0..dimensions)
                .map(|i| ((i % 19) as f32 - 8.) / 7.)
                .collect::<Vec<_>>();
            let strided = |v: &[f32]| {
                Array::from_slice(v, &[2, dimensions, 2])
                    .transpose_axes(&[0, 2, 1], &stream)
                    .unwrap()
            };
            let input = strided(&values);
            let gate = strided(&gates);
            let weight = Array::from_slice(&weights, &[dimensions]);
            let activated = gate
                .multiply(safemlx::ops::sigmoid(&gate, &stream).unwrap(), &stream)
                .unwrap();
            let gated = input.multiply(activated, &stream).unwrap();
            let grouped = gated.reshape(&[-1, groups, width], &stream).unwrap();
            let variance = grouped
                .square(&stream)
                .unwrap()
                .mean_axis(-1, true, &stream)
                .unwrap();
            let scale = variance
                .add(Array::from_f32(0.00001), &stream)
                .unwrap()
                .rsqrt(&stream)
                .unwrap();
            let expected = grouped
                .multiply(scale, &stream)
                .unwrap()
                .reshape(input.shape(), &stream)
                .unwrap()
                .multiply(&weight, &stream)
                .unwrap();
            same_bits(
                grouped_rms(&input, &gate, &weight, groups, 0.00001, &stream)
                    .unwrap()
                    .unwrap(),
                expected,
            );
        }
    }

    #[test]
    fn gelu_product_preserves_native_multiplication_tanh_and_exceptional_values() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let mut values = (0..18436)
            .map(|i| ((i * 37 % 8191) as f32 - 4095.) / 1024.)
            .collect::<Vec<_>>();
        for (pair, value) in values[..16].chunks_exact_mut(2).zip([
            0.,
            -0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-30,
            -1e-30,
            100.,
        ]) {
            pair.copy_from_slice(&[value, 1.25]);
        }
        let gate = Array::from_slice(&values, &[2, 4609, 2])
            .try_index_device((.., .., 0), &stream)
            .unwrap();
        let up = Array::from_slice(&values, &[2, 4609, 2])
            .try_index_device((.., .., 1), &stream)
            .unwrap();
        let expected = super::super::layers::gelu_approximate(&gate, &stream)
            .unwrap()
            .multiply(&up, &stream)
            .unwrap();
        same_bits(
            gelu_product(&gate, &up, &stream).unwrap().unwrap(),
            expected,
        );
        assert!(gelu_product(
            &gate.as_dtype(Dtype::Float16, &stream).unwrap(),
            &up,
            &stream
        )
        .unwrap()
        .is_none());
    }
}
