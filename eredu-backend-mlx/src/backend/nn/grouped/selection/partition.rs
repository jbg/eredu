//! Device-side compatibility repair for value-only cutoff ties.
use safemlx::{error::Exception, Array, Stream};

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
                    device const float* values = scores + row*WIDTH;
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
