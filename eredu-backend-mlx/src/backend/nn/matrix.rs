//! Reduced-precision row projections with a complete FP32 reduction.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::Dtype;
use safemlx::{error::Exception, Array, Stream};

/// Supplies the native row-dot mechanism when its dtype and device apply.
/// Weights remain row-major, including the leading bank axis for selected rows.
pub(crate) fn bf16_row_projection(
    input: &Array,
    weight: &Array,
    groups: Option<&Array>,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    bf16_projection(input, weight, groups, false, stream)
}

fn bf16_projection(
    input: &Array,
    weight: &Array,
    groups: Option<&Array>,
    columns: bool,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        use safemlx::{
            fast::{CustomKernelConfig, MetalKernel},
            DeviceType,
        };
        use std::cell::RefCell;
        thread_local! {
            static KERNEL: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
        }
        let width = input.dim(-1);
        if input.dtype() != Dtype::Bfloat16
            || weight.dtype() != Dtype::Bfloat16
            || (!columns && width % 32 != 0)
            || width == 0
            || stream.get_device()?.get_type()? != DeviceType::Gpu
        {
            return Ok(None);
        }
        let rows = input.size() / width as usize;
        let outputs = weight.dim(-2);
        if rows == 0 {
            return Ok(None);
        }
        let rows =
            i32::try_from(rows).map_err(|_| Exception::custom("matrix row count overflow"))?;
        if weight.dim(-1) != width || weight.ndim() != if groups.is_some() { 3 } else { 2 } {
            return Err(Exception::custom("invalid row projection geometry"));
        }
        if let Some(ids) = groups {
            if ids.ndim() != 1
                || ids.dim(0) != rows
                || !matches!(
                    ids.dtype(),
                    Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
                )
            {
                return Err(Exception::custom("invalid row projection group indices"));
            }
            let valid = ids
                .ge(Array::from_int(0), stream)?
                .logical_and(ids.lt(Array::from_int(weight.dim(0)), stream)?, stream)?;
            if !valid.all(None, stream)?.try_item::<bool>(stream)? {
                return Err(Exception::custom(
                    "row projection group index is outside the bank",
                ));
            }
        }
        let ids = groups
            .cloned()
            .unwrap_or_else(|| Array::from_slice(&[0i32], &[1]));
        let ids = ids.as_dtype(Dtype::Int32, stream)?;
        let config = CustomKernelConfig::new()
            .with_template_arg_int("WIDTH", width)
            .with_template_arg_int("COLUMNS", i32::from(columns))
            .with_template_arg_int("OUTPUTS", outputs)
            .with_template_arg_int("GROUPED", i32::from(groups.is_some()))
            .with_grid([32, outputs, rows])
            .with_thread_group([32, 1, 1])
            .with_output_arg([rows, outputs], Dtype::Bfloat16);
        let mut output = KERNEL.with(|cell| -> Result<Vec<Array>, Exception> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(MetalKernel::new(
                    "bf16_row_projection", ["input", "weight", "groups"], ["output"],
                    concat!(
                        "uint lane = thread_position_in_grid.x;",
                        "uint col = thread_position_in_grid.y;",
                        "uint row = thread_position_in_grid.z;",
                        "uint group = GROUPED ? uint(groups[row]) : 0;",
                        "size_t base = (size_t(group) * OUTPUTS + col) * WIDTH;",
                        "float sum = 0.0f;",
                        "if (COLUMNS) {",
                        " if(lane<4) { for(uint k=lane; k+3-lane<WIDTH; k+=4) sum+=float(input[size_t(row)*WIDTH+k])*float(weight[base+k]); }",
                        " if(lane==0) { for(uint k=(WIDTH/4)*4; k<WIDTH; ++k) sum+=float(input[size_t(row)*WIDTH+k])*float(weight[base+k]); }",
                        " float s1=simd_shuffle(sum,1), s2=simd_shuffle(sum,2), s3=simd_shuffle(sum,3);",
                        " if(lane==0) output[size_t(row)*OUTPUTS+col]=bfloat16_t(((sum+s1)+s2)+s3);",
                        " return; }",
                        "for (uint k = lane; k < WIDTH; k += 32) {",
                        " sum += float(input[size_t(row) * WIDTH + k]) * float(weight[base + k]);",
                        "}",
                        "sum += simd_shuffle_down(sum, 16);",
                        "sum += simd_shuffle_down(sum, 8);",
                        "sum += simd_shuffle_down(sum, 4);",
                        "float s1 = simd_shuffle(sum, 1);",
                        "float s2 = simd_shuffle(sum, 2);",
                        "float s3 = simd_shuffle(sum, 3);",
                        "if (lane == 0) output[size_t(row) * OUTPUTS + col] = bfloat16_t((sum + s1) + (s2 + s3));"
                    ), "", true, false,
                )?);
            }
            cell.borrow().as_ref().expect("row projection kernel initialized")
                .apply_device([input, weight, &ids], &config, stream)
        })?;
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("matrix input rank") = outputs;
        return output
            .pop()
            .expect("row projection output")
            .reshape(&shape, stream)
            .map(Some);
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = (input, weight, groups, columns, stream);
        Ok(None)
    }
}

/// Complete FP32 batched BF16 products with row-dot or strided-column sums.
pub(crate) fn bf16_batched_product(
    lhs: &Array,
    rhs: &Array,
    columns: bool,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if !cfg!(all(feature = "metal", not(feature = "cuda")))
        || lhs.size() == 0
        || rhs.size() == 0
        || lhs.dtype() != safemlx::Dtype::Bfloat16
        || rhs.dtype() != safemlx::Dtype::Bfloat16
        || lhs.ndim() != 4
        || rhs.ndim() != 4
        || lhs.shape()[..2] != rhs.shape()[..2]
        || lhs.dim(3) != rhs.dim(2)
        || stream.get_device()?.get_type()? != safemlx::DeviceType::Gpu
    {
        return Ok(None);
    }
    let batches = lhs
        .dim(0)
        .checked_mul(lhs.dim(1))
        .ok_or_else(|| Exception::custom("matrix batch count overflow"))?;
    let rows = lhs.dim(2);
    let width = lhs.dim(3);
    let outputs = rhs.dim(3);
    let input = lhs.reshape(&[-1, width], stream)?;
    let weights = rhs
        .swap_axes(-1, -2, stream)?
        .reshape(&[batches, outputs, width], stream)?;
    let ids = (0..batches)
        .flat_map(|batch| std::iter::repeat_n(batch, rows as usize))
        .collect::<Vec<_>>();
    let groups = Array::from_slice(&ids, &[ids.len() as i32]);
    bf16_projection(&input, &weights, Some(&groups), columns, stream)?
        .map(|output| output.reshape(&[lhs.dim(0), lhs.dim(1), rows, outputs], stream))
        .transpose()
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::{Device, DeviceType};
    #[test]
    fn bf16_native_row_projections_match_independent_dense_and_selected_dots() {
        let weights = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let dense = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/bf16_matmul.safetensors"
            ),
            &weights,
        )
        .unwrap();
        let grouped = Array::load_safetensors(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/validation/bf16_grouped.safetensors"
            ),
            &weights,
        )
        .unwrap();
        for width in [128, 768, 2560] {
            for (fixture, ids, expected) in [
                (&dense, None, format!("{width}.output.3")),
                (&grouped, Some(&grouped["ids"]), format!("{width}.output")),
            ] {
                let output = bf16_row_projection(
                    &fixture[&format!("{width}.input")],
                    &fixture[&format!("{width}.weight")],
                    ids,
                    &stream,
                )
                .unwrap()
                .unwrap();
                let output = output
                    .as_dtype(Dtype::Float32, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap();
                let expected = fixture[&expected]
                    .as_dtype(Dtype::Float32, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap();
                let differences = output
                    .as_slice::<f32>()
                    .iter()
                    .zip(expected.as_slice::<f32>())
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(differences, 0, "width {width}, grouped={}", ids.is_some());
            }
        }
    }
}
