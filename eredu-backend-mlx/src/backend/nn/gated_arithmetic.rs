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
