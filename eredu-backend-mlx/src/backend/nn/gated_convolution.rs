use safemlx::{error::Exception, Array, Stream};

#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::{
    fast::{CustomKernelConfig, MetalKernel},
    DeviceType, Dtype,
};

pub(crate) fn causal_depthwise_step(
    projected: &Array,
    history: &Array,
    weight: &Array,
    stream: &Stream,
) -> Result<Option<(Array, Array)>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use std::cell::RefCell;
        thread_local! {
            static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }

        let [batch, 1, width] = *projected.shape() else {
            return Ok(None);
        };
        let [channels, 1, taps] = *weight.shape() else {
            return Ok(None);
        };
        if batch <= 0
            || channels <= 0
            || channels.checked_mul(3) != Some(width)
            || taps <= 1
            || history.shape() != [batch, taps - 1, channels]
            || [projected, history, weight]
                .iter()
                .any(|array| array.dtype() != Dtype::Bfloat16)
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }

        let config = CustomKernelConfig::new()
            .with_template_arg_int("CHANNELS", channels)
            .with_template_arg_int("TAPS", taps)
            .with_grid([channels, batch, 1])
            .with_thread_group([128, 1, 1])
            .with_output_arg(&[batch, 1, channels], Dtype::Bfloat16)
            .with_output_arg(history.shape(), Dtype::Bfloat16);
        return KERNEL.with(|cell| {
            let mut kernel = cell.borrow_mut();
            if kernel.is_none() {
                *kernel = Some(MetalKernel::new(
                    "bf16_gated_causal_depthwise_step",
                    ["projected", "history", "weights"],
                    ["output", "next_history"],
                    include_str!("bf16_gated_causal_depthwise_step.metal"),
                    "",
                    true,
                    false,
                )?);
            }
            let outputs = kernel
                .as_ref()
                .expect("gated convolution kernel initialized")
                .apply_device([projected, history, weight], &config, stream)?;
            let [output, next_history] = outputs
                .try_into()
                .map_err(|_| Exception::custom("gated convolution requires two outputs"))?;
            Ok(Some((output, next_history)))
        });
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (projected, history, weight, stream);
        Ok(None)
    }
}
