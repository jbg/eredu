//! Device-side compatibility repair for value-only cutoff ties.
use safemlx::{error::Exception, Array, Stream};

#[path = "partial.rs"]
mod partial;

pub(super) fn repair(
    descending: &Array,
    indices: &Array,
    tied: &Array,
    count: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            Dtype,
        };
        use std::cell::RefCell;
        // libc++ sorts this small prefix with stable insertion/sorting networks.
        // Larger prefixes and banks retain the general compatibility path.
        if count > 16 || descending.dim(-1) > 1024 {
            return Ok(None);
        }
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let rows = descending.dim(0);
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", descending.dim(-1))
            .with_template_arg_int("COUNT", count)
            .with_template_arg_int("ROWS", rows)
            .with_grid([rows, 1, 1])
            .with_thread_group([1, 1, 1])
            .with_output_arg([rows, count], Dtype::Uint32);
        let repaired = KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "routing_value_partition",
                    ["scores", "indices", "tied"],
                    ["out"],
                    r#"
                    uint row = thread_position_in_grid.x;
                    if (row >= ROWS) return;
                    if (!tied) {
                        for (int i=0; i<COUNT; ++i) out[row*COUNT+i] = indices[row*COUNT+i];
                        return;
                    }
                    thread uint ids[WIDTH];
                    for (int i=0; i<WIDTH; ++i) ids[i] = i;
                    auto values = scores + row*WIDTH;
                    route_partition(values, ids, COUNT-1, WIDTH);
                    for (int i=0; i<COUNT; ++i) out[row*COUNT+i]=ids[i];
                    "#,
                    include_str!("partition.metal"),
                    true,
                    false,
                )?);
            }
            cell.borrow()
                .as_ref()
                .expect("partition initialized")
                .apply_one_device([descending, indices, tied], &config, stream)
        })?;
        use safemlx::ops::{
            argsort_axis, concatenate_axis,
            indexing::{take_along_axis, TryIndexOp},
        };
        if count <= 1 {
            return Ok(Some(repaired));
        }
        let prefix = repaired.try_index_device((.., ..count - 1), stream)?;
        let values = take_along_axis(descending, &prefix, -1, stream)?;
        let order = argsort_axis(&values, -1, stream)?;
        let prefix = take_along_axis(&prefix, &order, -1, stream)?;
        let last = repaired.try_index_device((.., count - 1..), stream)?;
        Ok(Some(concatenate_axis(&[&prefix, &last], -1, stream)?))
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (descending, indices, tied, count, stream);
        Ok(None)
    }
}

/// Retains the value-only median partition without MLX's prefix sorting.
/// The native Metal path shares the cutoff-repair partition kernel; other
/// devices use bounded score and index rows through the same host algorithm.
pub(super) fn unranked(scores: &Array, count: i32, stream: &Stream) -> Result<Array, Exception> {
    use safemlx::{Device, DeviceType, Dtype};
    if scores.ndim() != 2 || count <= 0 || count > scores.dim(1) {
        return Err(Exception::custom("invalid unranked selector geometry"));
    }
    if !matches!(
        scores.dtype(),
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(Exception::custom(
            "median partition requires F32/F16/BF16 selection scores",
        ));
    }
    let descending = scores
        .as_dtype(Dtype::Float32, stream)?
        .multiply(Array::from_f32(-1.0), stream)?;
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if stream.get_device()?.get_type()? == DeviceType::Gpu && count <= 16 && scores.dim(1) <= 1024 {
        use safemlx::fast::{CustomKernelConfig, MetalKernel};
        use std::cell::RefCell;
        thread_local! { static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) }; }
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", scores.dim(1))
            .with_template_arg_int("COUNT", count)
            .with_template_arg_int("ROWS", scores.dim(0))
            .with_grid([scores.dim(0), 1, 1])
            .with_thread_group([1, 1, 1])
            .with_output_arg([scores.dim(0), count], Dtype::Uint32);
        return KERNEL.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new("unranked_value_partition", ["scores"], ["out"],
                    "uint row=thread_position_in_grid.x; if(row>=ROWS) return; thread uint ids[WIDTH]; for(int i=0;i<WIDTH;++i) ids[i]=i; route_partition(scores+row*WIDTH,ids,COUNT-1,WIDTH); for(int i=0;i<COUNT;++i) out[row*COUNT+i]=ids[i];",
                    include_str!("partition.metal"), true, false)?);
            }
            cell.borrow().as_ref().expect("unranked partition initialized").apply_one_device([&descending], &config, stream)
        });
    }
    let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let host = descending
        .copy(&cpu)?
        .contiguous(false, &cpu)?
        .into_evaluated()?;
    let width = scores.dim(1) as usize;
    let mut output = Vec::with_capacity(scores.dim(0) as usize * count as usize);
    let mut ids = (0..width).collect::<Vec<_>>();
    for row in host.as_slice::<f32>().chunks_exact(width) {
        for (index, id) in ids.iter_mut().enumerate() {
            *id = index;
        }
        partial::partition_ids(row, &mut ids, count as usize - 1);
        output.extend(ids[..count as usize].iter().map(|&id| id as i32));
    }
    Array::from_slice(&output, &[scores.dim(0), count]).copy(stream)
}
