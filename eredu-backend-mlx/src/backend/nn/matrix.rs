//! Reduced-precision row projections with a complete FP32 reduction.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use safemlx::Dtype;
use safemlx::{Array, Stream, error::Exception};

// The BF16 alternative uses the same fixed invocation and source family as
// ordinary row projection. Source/cache birth belongs to its shared initializer.
pub(crate) fn grouped_projection_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes(
    )?;
    let safe = crate::backend::managed_memory::bf16_projection_kernel::control_bytes()?;
    [
        // Final shape storage is local and fixed; wider original ranks refuse
        // before ID validation/casts or kernel construction.
        size_of::<[i32; 32]>(),
        size_of::<[i32; 3]>(),
        size_of::<[&Array; 3]>(),
        size_of::<[Option<Array>; 2]>(),
        size_of::<Array>() * 2,
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<usize>() * 3,
        size_of::<i32>() * 3,
        size_of::<bool>() * 2,
        size_of::<Option<safemlx::OriginalScopeObserver>>(),
        // grouped_matmul and its candidate each inspect the scalar kind.
        Stream::device_type_control_bytes()?.checked_mul(2)?,
    ]
    .into_iter()
    .try_fold(safe, usize::checked_add)
}

// The two batched attention products and any query/key tiles call this
// producer sequentially. Only one fixed producer frame is live at a time;
// numerical IDs are filled in their single final native backing.
pub(crate) fn batched_input_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<safemlx::RepeatedI32InputPlan>(),
        size_of::<Option<safemlx::RepeatedI32InputPlan>>(),
        size_of::<[i32; 4]>(),
        size_of::<i32>() * 4,
        size_of::<usize>() * 2,
    ]
    .into_iter()
    .try_fold(
        safemlx::RepeatedI32InputPlan::control_bytes()?,
        usize::checked_add,
    )
}

/// Shared geometry predicate for ordinary row projection and its source-qualified
/// workspace recipe. Grouped column products retain their different contract.
pub(crate) const fn bf16_row_width_supported(width: i32) -> bool {
    width > 0 && width % 32 == 0
}

/// Fixed controls of the shared row-projection probe when it rejects either
/// a non-BF16 input or a CPU stream. No shader or invocation source is selected
/// by that rejection; the subsequent Matmul has its own native population.
pub(crate) fn row_projection_probe_control_bytes() -> Option<usize> {
    use std::mem::{size_of,size_of_val};
    let parts=[size_of::<(&Array,&Array,Option<&Array>,&Stream)>(),
        size_of::<(&Array,&Array,Option<&Array>,bool,&Stream)>(),
        size_of::<Result<Option<Array>,Exception>>()*2,
        size_of::<Option<Array>>(),size_of::<Array>()*3,
        size_of::<[i32;32]>(),size_of::<Vec<i32>>(),
        size_of::<i32>()*3,size_of::<usize>(),size_of::<bool>()*2,
        size_of::<safemlx::Dtype>()*2,size_of::<safemlx::DeviceType>(),
        size_of::<Result<safemlx::DeviceType,Exception>>()];
    parts.into_iter().try_fold(Stream::device_type_control_bytes()?.checked_add(size_of_val(&parts))?,usize::checked_add)
}

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
        use crate::backend::managed_memory::bf16_projection_kernel as kernel;
        use safemlx::DeviceType;
        let width = input.dim(-1);
        if input.dtype() != Dtype::Bfloat16
            || weight.dtype() != Dtype::Bfloat16
            || (!columns && !bf16_row_width_supported(width))
            || width == 0
            || stream.device_type()? != DeviceType::Gpu
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
        kernel::validate_call(input.ndim())?;
        let mut prepared_groups = None;
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
            prepared_groups = super::tensor::original_group_indices(ids, weight.dim(0), stream)?;
            if prepared_groups.is_none() {
                let valid = ids
                    .ge(Array::try_from_int(0)?, stream)?
                    .logical_and(ids.lt(Array::try_from_int(weight.dim(0))?, stream)?, stream)?;
                if !valid.all(None, stream)?.try_item::<bool>(stream)? {
                    return Err(Exception::custom(
                        "row projection group index is outside the bank",
                    ));
                }
            }
        }
        let ids = match prepared_groups {
            Some(ids) => ids,
            None => match groups {
                Some(ids) => ids.clone(),
                None => Array::try_from_slice(&[0i32], &[1])?,
            },
        };
        let ids = ids.as_dtype(Dtype::Int32, stream)?;
        let output = kernel::apply(
            [input, weight, &ids],
            rows,
            outputs,
            columns,
            groups.is_some(),
            stream,
        )?;
        if input.ndim() <= kernel::OUTPUT_DIMENSIONS {
            let mut shape = [0; kernel::OUTPUT_DIMENSIONS];
            shape[..input.ndim()].copy_from_slice(input.shape());
            shape[input.ndim() - 1] = outputs;
            return output.reshape(&shape[..input.ndim()], stream).map(Some);
        }
        // Ordinary inputs retain the existing unrestricted rank behavior.
        // Original rank validation above rejects before any new native work.
        let mut shape = input.shape().to_vec();
        *shape.last_mut().expect("matrix input rank") = outputs;
        return output.reshape(&shape, stream).map(Some);
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
        || stream.device_type()? != safemlx::DeviceType::Gpu
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
    // Validate the exact final I32 extent before creating either view. This
    // replaces the host Vec and unchecked length cast, retaining the same
    // batch-major sequence and one eager native source generation.
    let ids = usize::try_from(batches)
        .ok()
        .zip(usize::try_from(rows).ok())
        .and_then(|(batches, rows)| safemlx::RepeatedI32InputPlan::new(batches, rows))
        .ok_or_else(|| Exception::custom("matrix group input extent overflow"))?;
    let input = lhs.reshape(&[-1, width], stream)?;
    let weights = rhs
        .swap_axes(-1, -2, stream)?
        .reshape(&[batches, outputs, width], stream)?;
    let groups = ids.create()?;
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
