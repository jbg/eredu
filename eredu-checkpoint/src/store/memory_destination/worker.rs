// Shared original in-memory selection equations; storage policy only.
use super::*;

pub(super) fn select<'a, S: Storage<'a>>(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
    selection: &'a TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
    mut storage: S,
) -> Result<(Range<usize>, Option<S::Buffer>), StoreError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok((0..data.len(), None));
    }
    let bits = dtype.bitsize();
    let scalar_bytes = bits.checked_div(8).filter(|_| bits.is_multiple_of(8));
    if let (
        Some(scalar_bytes),
        TensorSelection::Contiguous {
            offset_elements,
            shape,
        },
    ) = (scalar_bytes, selection)
    {
        let start =
            offset_elements
                .checked_mul(scalar_bytes)
                .ok_or_else(|| StoreError::Overflow {
                    context: format!("contiguous byte start for {key:?}"),
                })?;
        let end = checked_elements(key, shape)?
            .checked_mul(scalar_bytes)
            .and_then(|length| start.checked_add(length))
            .ok_or_else(|| StoreError::Overflow {
                context: format!("contiguous byte end for {key:?}"),
            })?;
        return data
            .get(start..end)
            .map(|_| (start..end, None))
            .ok_or_else(|| invalid_selection(key, "contiguous byte span outside payload"));
    }
    if let (
        Some(_),
        TensorSelection::Range {
            axis: 0,
            start,
            end,
        },
    ) = (scalar_bytes, selection)
    {
        let row_bytes = data
            .len()
            .checked_div(shape[0])
            .filter(|_| data.len().is_multiple_of(shape[0]))
            .ok_or_else(|| invalid_selection(key, "payload is not row divisible"))?;
        let start = start * row_bytes;
        let end = end * row_bytes;
        return Ok((start..end, None));
    }
    if matches!(policy, ReadPolicy::AllowFullTensorRead) {
        return Ok((0..data.len(), None));
    }
    let (axis, indices): (usize, Indices<'a>) = match selection {
        TensorSelection::Range { axis, start, end } => (*axis, storage.range(*start..*end)),
        TensorSelection::Indices { axis, indices } => (*axis, storage.indices(indices)),
        TensorSelection::Contiguous { .. } => {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "packed contiguous selection is not byte aligned".into(),
            });
        }
        TensorSelection::Full => unreachable!(),
    };
    let axis_len = shape[axis];
    let outer = shape[..axis].iter().product::<usize>();
    let inner = shape[axis + 1..].iter().product::<usize>();
    let output_bits = checked_elements(key, output_shape)?
        .checked_mul(bits)
        .ok_or_else(|| StoreError::Overflow {
            context: format!("selected bit length for {key:?}"),
        })?;
    if !output_bits.is_multiple_of(8) {
        return Err(StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "selected packed payload is not byte aligned".into(),
        });
    }
    let mut output = storage.output(output_bits / 8)?;
    if bits == 4 {
        if !inner.is_multiple_of(2)
            || indices
                .iter()
                .any(|index| !(index * inner).is_multiple_of(2))
        {
            return Err(StoreError::BoundedSelectionUnavailable {
                key: key.into(),
                message: "FP4 selection crosses a nibble boundary".into(),
            });
        }
        let block_bytes = inner / 2;
        for outer_index in 0..outer {
            for index in indices.iter() {
                let start = (outer_index * axis_len + index) * block_bytes;
                S::append(
                    &mut output,
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                )?;
            }
        }
    } else {
        let scalar_bytes = scalar_bytes.ok_or_else(|| StoreError::BoundedSelectionUnavailable {
            key: key.into(),
            message: "stored scalar width is not byte aligned".into(),
        })?;
        let block_bytes = inner * scalar_bytes;
        for outer_index in 0..outer {
            for index in indices.iter() {
                let start = (outer_index * axis_len + index) * block_bytes;
                S::append(
                    &mut output,
                    data.get(start..start + block_bytes)
                        .ok_or_else(|| invalid_selection(key, "selection exceeds payload"))?,
                )?;
            }
        }
    }
    Ok((0..S::len(&output), Some(output)))
}
