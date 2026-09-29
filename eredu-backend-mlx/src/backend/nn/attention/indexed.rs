//! Bounded selected-position attention. One query's selected rows are live at a
//! time; gathering never broadcasts or copies the full history per query.
use crate::MlxTensor;
use eredu_nn::IndexedAttentionInput;
use safemlx::{
    error::Exception,
    ops::{
        broadcast_to, concatenate_axis,
        indexing::{take_axis, TryIndexOp},
        r#where, zeros_dtype,
    },
    Array, Dtype, Stream,
};

fn positions(
    input: &IndexedAttentionInput<'_, MlxTensor>,
    batch: i32,
    query: i32,
    stream: &Stream,
) -> Result<Vec<Option<i32>>, Exception> {
    let array = input
        .selected_positions
        .as_array()
        .try_index_device((batch, query, ..), stream)?;
    let values = array.evaluated()?;
    let raw: Vec<i64> = match array.dtype() {
        Dtype::Int32 => values
            .as_slice::<i32>()
            .iter()
            .map(|v| i64::from(*v))
            .collect(),
        Dtype::Uint32 => values
            .as_slice::<u32>()
            .iter()
            .map(|v| i64::from(*v))
            .collect(),
        _ => {
            return Err(Exception::custom(
                "indexed positions must be 32-bit integers",
            ))
        }
    };
    let valid = input
        .validity
        .map(|value| {
            if value.as_array().dtype() != Dtype::Bool {
                return Err(Exception::custom("indexed validity must be boolean"));
            }
            let row = value
                .as_array()
                .try_index_device((batch, query, ..), stream)?;
            Ok(row.evaluated()?.as_slice::<bool>().to_vec())
        })
        .transpose()?;
    raw.into_iter()
        .enumerate()
        .map(|(slot, position)| {
            if position == -1 || valid.as_ref().is_some_and(|v| !v[slot]) {
                return Ok(None);
            }
            i32::try_from(position)
                .ok()
                .filter(|v| *v >= 0)
                .map(Some)
                .ok_or_else(|| {
                    Exception::custom("indexed position is outside signed 32-bit token geometry")
                })
        })
        .collect()
}

fn mask_row(
    mask: Option<&MlxTensor>,
    input: &IndexedAttentionInput<'_, MlxTensor>,
    batch: i32,
    query: i32,
    tokens: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let q = input.queries.as_array();
    let shape = [q.dim(0), q.dim(1), q.dim(2), tokens];
    let Some(mask) = mask else {
        return Array::zeros::<f32>(&[1, q.dim(1), 1, tokens], stream);
    };
    let mask = broadcast_to(mask.as_array(), &shape, stream)?
        .try_index_device((batch..batch + 1, .., query..query + 1, ..), stream)?;
    if mask.dtype() == Dtype::Bool {
        r#where(
            mask,
            Array::from_f32(0.0),
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )
    } else {
        mask.as_dtype(Dtype::Float32, stream)
    }
}

/// Runs selected attention with resident grouped-query source tensors.
pub fn indexed_sparse_attention(
    input: &IndexedAttentionInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<Array, Exception> {
    if input.queries.as_array().shape().get(2) == Some(&1)
        && stream.get_device()?.get_type()? == safemlx::DeviceType::Gpu
    {
        return resident_decode(input, stream);
    }
    indexed_attention_with_reader(input, stream, |batch, positions, stream| {
        let origin = i64::from(input.key_position_offset);
        let length = i64::from(input.keys.as_array().dim(2));
        let indexes = positions
            .iter()
            .map(|position| {
                let index = i64::from(*position) - origin;
                if index < 0 || index >= length {
                    return Err(Exception::custom(
                        "indexed position is outside retained source",
                    ));
                }
                Ok(index as i32)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let indexes = Array::from_slice(&indexes, &[indexes.len() as i32]);
        let gather = |source: &MlxTensor| {
            let batch = source
                .as_array()
                .try_index_device((batch..batch + 1, .., .., ..), stream)?;
            take_axis(batch, &indexes, 2, stream)
        };
        Ok((gather(input.keys)?, gather(input.values)?))
    })
}

// One resident query per batch needs no host position list or reader lease.
// Keep indices and validity on-device and retain assertions until completion.
fn resident_decode(
    input: &IndexedAttentionInput<'_, MlxTensor>,
    stream: &Stream,
) -> Result<Array, Exception> {
    input
        .validate()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let q = input.queries.as_array();
    let source_keys = input.keys.as_array();
    let source_values = input.values.as_array();
    let length = source_keys.dim(2);
    let selected = input.selected_positions.as_array().dim(2);
    if !matches!(
        input.selected_positions.as_array().dtype(),
        Dtype::Int32 | Dtype::Uint32
    ) || input
        .validity
        .is_some_and(|v| v.as_array().dtype() != Dtype::Bool)
    {
        return Err(Exception::custom(
            "indexed positions/validity must be 32-bit integers/boolean",
        ));
    }
    let mut outputs = Vec::new();
    for batch in 0..q.dim(0) {
        let positions = input
            .selected_positions
            .as_array()
            .try_index_device((batch, 0, ..), stream)?
            .as_dtype(Dtype::Int64, stream)?;
        let mut valid = positions.ne(Array::from_int(-1), stream)?;
        if let Some(validity) = input.validity {
            valid = valid.logical_and(
                validity
                    .as_array()
                    .try_index_device((batch, 0, ..), stream)?,
                stream,
            )?;
        }
        let indices = positions.subtract(Array::from_int(input.key_position_offset), stream)?;
        let in_range = indices
            .ge(Array::from_int(0), stream)?
            .logical_and(indices.lt(Array::from_int(length), stream)?, stream)?;
        super::super::tensor::register_device_validation(
            valid
                .logical_and(in_range.logical_not(stream)?, stream)?
                .any(false, stream)?,
            "indexed position is outside retained source",
        )?;
        valid = valid.logical_and(in_range, stream)?;
        let safe = r#where(&valid, &indices, Array::from_int(0), stream)?
            .as_dtype(Dtype::Int32, stream)?;
        let row_valid = valid.reshape(&[1, 1, selected, 1], stream)?;
        let gather = |source: &Array| -> Result<Array, Exception> {
            if length == 0 {
                return zeros_dtype(
                    &[1, source.dim(1), selected, source.dim(3)],
                    source.dtype(),
                    stream,
                );
            }
            let rows = take_axis(
                source.try_index_device((batch..batch + 1, .., .., ..), stream)?,
                &safe,
                2,
                stream,
            )?;
            r#where(&row_valid, rows, Array::from_f32(0.), stream)?.as_dtype(source.dtype(), stream)
        };
        let keys = gather(source_keys)?;
        let values = gather(source_values)?;
        let mask = r#where(
            valid.reshape(&[1, 1, 1, selected], stream)?,
            mask_row(input.mask, input, batch, 0, selected, stream)?,
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )?;
        outputs.push(attend_selected(
            input, batch, 0, keys, values, mask, stream,
        )?);
    }
    concatenate_axis(&outputs, 0, stream)
}

fn attend_selected(
    input: &IndexedAttentionInput<'_, MlxTensor>,
    batch: i32,
    query: i32,
    selected_keys: Array,
    selected_values: Array,
    selected_mask: Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let q = input.queries.as_array();
    let (keys, values, mask) = if let Some(local) = &input.local {
        let local_keys = local
            .keys
            .as_array()
            .try_index_device((batch..batch + 1, .., .., ..), stream)?;
        let local_values = local
            .values
            .as_array()
            .try_index_device((batch..batch + 1, .., .., ..), stream)?;
        let local_mask = mask_row(local.mask, input, batch, query, local_keys.dim(2), stream)?;
        (
            concatenate_axis(&[local_keys, selected_keys], 2, stream)?,
            concatenate_axis(&[local_values, selected_values], 2, stream)?,
            concatenate_axis(&[local_mask, selected_mask], -1, stream)?,
        )
    } else {
        (selected_keys, selected_values, selected_mask)
    };
    let query = q.try_index_device((batch..batch + 1, .., query..query + 1, ..), stream)?;
    if keys.dim(2) == 0 {
        return zeros_dtype(&[1, q.dim(1), 1, values.dim(3)], q.dtype(), stream);
    }
    let mut output = super::attention_with_softcap(
        &query,
        &keys,
        &values,
        input.scale,
        Some(&mask),
        input.sinks.map(MlxTensor::as_array),
        None,
        input.arithmetic,
        stream,
    )?;
    if input.sinks.is_none() {
        let any = mask
            .gt(Array::from_f32(f32::NEG_INFINITY), stream)?
            .any_axis(-1, true, stream)?;
        output = r#where(any, output, Array::from_f32(0.), stream)?.as_dtype(q.dtype(), stream)?;
    }
    Ok(output)
}

/// Executes through a cache-owned reader. The reader receives sorted unique
/// absolute positions for one batch/query, returns exactly those rows, and must
/// complete copies before releasing any source leases. Results are reordered to
/// the requested slots here, including duplicates and invalid padding.
pub(crate) fn indexed_attention_with_reader(
    input: &IndexedAttentionInput<'_, MlxTensor>,
    stream: &Stream,
    mut read: impl FnMut(i32, &[i32], &Stream) -> Result<(Array, Array), Exception>,
) -> Result<Array, Exception> {
    input
        .validate()
        .map_err(|e| Exception::custom(e.to_string()))?;
    let q = input.queries.as_array();
    let keys = input.keys.as_array();
    let values = input.values.as_array();
    let selected = input.selected_positions.as_array().dim(2);
    let mut batches = Vec::new();
    for batch in 0..q.dim(0) {
        let mut outputs = Vec::new();
        for query in 0..q.dim(2) {
            let positions = positions(input, batch, query, stream)?;
            let mut unique = positions.iter().flatten().copied().collect::<Vec<_>>();
            unique.sort_unstable();
            unique.dedup();
            let (selected_keys, selected_values) = if unique.is_empty() {
                (
                    zeros_dtype(
                        &[1, keys.dim(1), selected, keys.dim(3)],
                        keys.dtype(),
                        stream,
                    )?,
                    zeros_dtype(
                        &[1, values.dim(1), selected, values.dim(3)],
                        values.dtype(),
                        stream,
                    )?,
                )
            } else {
                let (gathered_keys, gathered_values) = read(batch, &unique, stream)?;
                if gathered_keys.shape() != [1, keys.dim(1), unique.len() as i32, keys.dim(3)]
                    || gathered_values.shape()
                        != [1, values.dim(1), unique.len() as i32, values.dim(3)]
                {
                    return Err(Exception::custom(
                        "selected history reader returned incorrect row geometry",
                    ));
                }
                let slots = positions
                    .iter()
                    .map(|position| {
                        position.map_or(0, |p| {
                            unique
                                .binary_search(&p)
                                .expect("selected position retained")
                                as i32
                        })
                    })
                    .collect::<Vec<_>>();
                let indexes = Array::from_slice(&slots, &[selected]);
                let valid = Array::from_slice(
                    &positions.iter().map(Option::is_some).collect::<Vec<_>>(),
                    &[1, 1, selected, 1],
                );
                // Invalid slots must not read a real row or retain NaNs from the
                // harmless gather index used to fill their physical positions.
                (
                    r#where(
                        &valid,
                        take_axis(gathered_keys, &indexes, 2, stream)?,
                        Array::from_f32(0.0),
                        stream,
                    )?
                    .as_dtype(keys.dtype(), stream)?,
                    r#where(
                        &valid,
                        take_axis(gathered_values, &indexes, 2, stream)?,
                        Array::from_f32(0.0),
                        stream,
                    )?
                    .as_dtype(values.dtype(), stream)?,
                )
            };
            let valid = Array::from_slice(
                &positions.iter().map(Option::is_some).collect::<Vec<_>>(),
                &[1, 1, 1, selected],
            );
            let selected_mask = r#where(
                valid,
                mask_row(input.mask, input, batch, query, selected, stream)?,
                Array::from_f32(f32::NEG_INFINITY),
                stream,
            )?;
            let output = attend_selected(
                input,
                batch,
                query,
                selected_keys,
                selected_values,
                selected_mask,
                stream,
            )?;
            // Complete the bounded gather/attention graph before retaining the
            // next query. Only completed output rows accumulate across a prefill.
            safemlx::transforms::eval([&output])?;
            outputs.push(output);
        }
        batches.push(concatenate_axis(&outputs, 2, stream)?);
    }
    concatenate_axis(&batches, 0, stream)
}
