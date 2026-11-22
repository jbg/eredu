use safemlx::{
    error::Exception,
    fast::{CustomKernelConfig, MetalKernel},
    Array, DeviceType, Dtype, Stream,
};
use std::cell::RefCell;

// One row of FP32 exponentials fits in threadgroup memory.
pub(super) const MAX_KEYS: i32 = 4096;

pub(super) fn apply(
    scores: &Array,
    scale: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let [_, _, _, keys] = scores.shape() else {
        return Ok(None);
    };
    if scores.dtype() != Dtype::Bfloat16
        || !(4..=MAX_KEYS).contains(keys)
        || scores.size() == 0
        || stream.get_device()?.get_type()? != DeviceType::Gpu
    {
        return Ok(None);
    }
    let rows = i32::try_from(scores.size() / *keys as usize)
        .map_err(|_| Exception::custom("attention score row count overflow"))?;

    let (mask, mask_mode) = if let Some(mask) = mask {
        let mode = match mask.dtype() {
            Dtype::Bool => 1,
            Dtype::Bfloat16 | Dtype::Float32 => 2,
            _ => return Ok(None),
        };
        if mask.ndim() > 4 {
            return Ok(None);
        }
        let mut shape = [1; 4];
        shape[4 - mask.ndim()..].copy_from_slice(mask.shape());
        if shape
            .iter()
            .zip(scores.shape())
            .any(|(&axis, &score_axis)| axis != 1 && axis != score_axis)
        {
            return Ok(None);
        }
        (mask.reshape(&shape, stream)?, mode)
    } else {
        (scores.clone(), 0)
    };

    let config = CustomKernelConfig::new()
        .with_template_arg_int("MASK_MODE", mask_mode)
        .with_template_arg_int("CAPACITY", (*keys as u32).next_power_of_two() as i32)
        .with_grid([32, rows, 1])
        .with_thread_group([32, 1, 1])
        .with_output_arg(scores.shape(), Dtype::Bfloat16);
    let shape = Array::from_slice(scores.shape(), &[4]);
    let scale = Array::from_slice(&[scale], &[1]);
    // Keep broadcast masks strided. Only the score matrix needs contiguous rows.
    let scores = scores.contiguous(false, stream)?;
    thread_local! {
        static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    }
    KERNEL.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(MetalKernel::new(
                "ordered_bf16_attention_softmax",
                ["scores", "mask", "dimensions", "scale"],
                ["probabilities"],
                include_str!("score_softmax.metal"),
                include_str!("../exp_f32.metal"),
                false,
                false,
            )?);
        }
        cell.borrow()
            .as_ref()
            .expect("attention softmax kernel initialized")
            .apply_one_device([&scores, &mask, &shape, &scale], &config, stream)
            .map(Some)
    })
}
