//! Native score operands for relative and selected-position attention.

use std::ops::Range;

use candle_core::{DType, Tensor as NativeTensor};
use eredu_nn::{
    AttentionRequest, Error, IndexedAttentionInput, NeuralBackend, RelativeAttentionInput, Tensor,
};

use crate::{CandleContext, CandleNeuralBackend, CandleTensor};

#[derive(Clone)]
pub(crate) struct RelativeScores {
    profiles: CandleTensor,
    query_start: i32,
    scaling: Option<NativeTensor>,
}

impl RelativeScores {
    pub(crate) fn prepare(
        input: &RelativeAttentionInput<'_, CandleTensor>,
        context: &CandleContext,
    ) -> Result<(CandleTensor, Self), Error> {
        input.validate()?;
        let rows = input.queries.dim(2);
        let scaling = (|| -> candle_core::Result<_> {
            if let (None, Some(floor)) = (input.window, input.log_scaling_floor) {
                let positions = (input.query_offset..input.query_offset + rows)
                    .map(|position| (position + 1) as f32)
                    .collect::<Vec<_>>();
                let ratio = NativeTensor::new(positions, &context.device)?
                    .affine(1. / f64::from(floor), 0.)?;
                let scale = ratio
                    .maximum(1f32)?
                    .log()?
                    .affine(f64::from(input.log_scaling_alpha), 1.)?;
                Ok(Some(scale.reshape((1, 1, rows as usize, 1))?))
            } else {
                Ok(None)
            }
        })()
        .map_err(Error::backend_source)?;
        let query = &input.queries.value;
        let scaled = query
            .to_dtype(DType::F32)
            .and_then(|query| match &scaling {
                Some(scaling) => query.broadcast_mul(scaling),
                None => Ok(query),
            })
            .and_then(|query| query.affine(1. / query.dims()[3] as f64, 0.));
        Ok((
            CandleTensor::wrap(scaled)?,
            Self {
                profiles: input.profiles.clone(),
                query_start: input.query_offset,
                scaling,
            },
        ))
    }

    pub(crate) fn prepared_bytes(&self, queries: &CandleTensor) -> usize {
        let bytes = |value: &NativeTensor| {
            value
                .elem_count()
                .saturating_mul(value.dtype().size_in_bytes())
        };
        bytes(&queries.value)
            .saturating_add(self.scaling.as_ref().map_or(0, bytes))
            .saturating_add(bytes(&self.profiles.value))
    }

    pub(crate) fn bias(&self, columns: Range<i64>) -> candle_core::Result<NativeTensor> {
        let (batch, heads, rows, extent) = self.profiles.value.dims4()?;
        let start = i64::from(self.query_start);
        let width = usize::try_from(columns.end - columns.start)?;
        let device = self.profiles.value.device();
        let cells = rows.checked_mul(width).ok_or_else(|| {
            candle_core::Error::Msg("relative bias page geometry overflows".into())
        })?;
        let mut indices = Vec::with_capacity(cells);
        let mut eligible = Vec::with_capacity(cells);
        for query in start..start + rows as i64 {
            for key in columns.clone() {
                let distance = query - key;
                let valid = (0..extent as i64).contains(&distance);
                indices.push(if valid { distance as u32 } else { 0 });
                eligible.push(u8::from(valid));
            }
        }
        let shape = (batch, heads, rows, width);
        let indices = NativeTensor::from_vec(indices, (1, 1, rows, width), device)?
            .broadcast_as(shape)?
            .contiguous()?;
        let eligible =
            NativeTensor::from_vec(eligible, (1, 1, rows, width), device)?.broadcast_as(shape)?;
        let profiles = self.profiles.value.to_dtype(DType::F32)?.contiguous()?;
        let bias = profiles.gather(&indices, 3)?;
        let zeros = NativeTensor::zeros((), DType::F32, device)?.broadcast_as(shape)?;
        let bias = eligible.where_cond(&bias, &zeros)?;
        match &self.scaling {
            Some(scaling) => bias.broadcast_mul(scaling),
            None => Ok(bias),
        }
    }
}

pub(crate) fn indexed_positions(
    input: &IndexedAttentionInput<'_, CandleTensor>,
    batch: i32,
    query: i32,
) -> Result<Vec<Option<i32>>, Error> {
    let row = |tensor: &CandleTensor| -> candle_core::Result<NativeTensor> {
        let (_, _, width) = tensor.value.dims3()?;
        tensor
            .value
            .narrow(0, usize::try_from(batch)?, 1)?
            .narrow(1, usize::try_from(query)?, 1)?
            .reshape(width)
    };
    let positions = (|| -> candle_core::Result<Vec<i64>> {
        let positions = row(input.selected_positions)?;
        match positions.dtype() {
            DType::I32 => Ok(positions
                .to_vec1::<i32>()?
                .into_iter()
                .map(i64::from)
                .collect()),
            DType::U32 => Ok(positions
                .to_vec1::<u32>()?
                .into_iter()
                .map(i64::from)
                .collect()),
            _ => candle_core::bail!("indexed positions must be 32-bit integers"),
        }
    })()
    .map_err(Error::backend_source)?;
    let valid = input
        .validity
        .map(|validity| row(validity)?.to_vec1::<u8>())
        .transpose()
        .map_err(Error::backend_source)?;
    eredu_nn::decode_indexed_positions(positions, |slot| {
        valid.as_ref().is_none_or(|mask| mask[slot] != 0)
    })
}

fn mask_row(
    mask: Option<&CandleTensor>,
    query_shape: (usize, usize, usize, usize),
    batch: usize,
    query: usize,
    width: usize,
    context: &CandleContext,
) -> candle_core::Result<NativeTensor> {
    let (batches, heads, rows, _) = query_shape;
    let shape = (1, heads, 1, width);
    let Some(mask) = mask else {
        return NativeTensor::zeros(shape, DType::F32, &context.device);
    };
    let mask = mask
        .value
        .broadcast_as((batches, heads, rows, width))?
        .narrow(0, batch, 1)?
        .narrow(2, query, 1)?;
    if mask.dtype() == DType::U8 {
        mask.where_cond(
            &NativeTensor::zeros((), DType::F32, &context.device)?.broadcast_as(shape)?,
            &NativeTensor::full(f32::NEG_INFINITY, (), &context.device)?.broadcast_as(shape)?,
        )
    } else {
        mask.to_dtype(DType::F32)
    }
}

pub(crate) fn attend_indexed_row(
    input: &IndexedAttentionInput<'_, CandleTensor>,
    batch: i32,
    query: i32,
    slots: &[Option<i32>],
    gathered: Option<[CandleTensor; 2]>,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    let q = &input.queries.value;
    let selected = input.selected_positions.dim(2) as usize;
    let batch = usize::try_from(batch).map_err(Error::backend_source)?;
    let query = usize::try_from(query).map_err(Error::backend_source)?;
    let out = (|| {
        let shape = q.dims4()?;
        let (_, heads, _, _) = shape;
        if slots.len() != selected || slots.iter().flatten().any(|&slot| slot < 0) {
            candle_core::bail!("invalid indexed attention slot map");
        }
        let mut keys = Vec::with_capacity(2);
        let mut values = Vec::with_capacity(2);
        let mut masks = Vec::with_capacity(2);
        if let Some(local) = &input.local {
            let width = local.keys.dim(2) as usize;
            if width != 0 {
                keys.push(local.keys.value.narrow(0, batch, 1)?);
                values.push(local.values.value.narrow(0, batch, 1)?);
                masks.push(mask_row(local.mask, shape, batch, query, width, context)?);
            }
        }
        if let Some([gathered_keys, gathered_values]) = &gathered {
            let unique = slots
                .iter()
                .flatten()
                .max()
                .map_or(0, |&slot| slot as usize + 1);
            let indices = NativeTensor::from_vec(
                slots
                    .iter()
                    .map(|slot| slot.unwrap_or(0) as u32)
                    .collect::<Vec<_>>(),
                selected,
                &context.device,
            )?;
            let flags = slots
                .iter()
                .map(|slot| u8::from(slot.is_some()))
                .collect::<Vec<_>>();
            let valid = NativeTensor::from_vec(flags, selected, &context.device)?;
            let restore = |source: &CandleTensor, rows: &CandleTensor| {
                let expected = (1, source.dim(1) as usize, unique, source.dim(3) as usize);
                if unique == 0
                    || rows.value.dims4()? != expected
                    || rows.value.dtype() != source.value.dtype()
                {
                    candle_core::bail!(
                        "selected history reader returned incorrect row geometry or dtype"
                    );
                }
                let rows = rows.value.index_select(&indices, 2)?;
                let valid = valid
                    .reshape((1, 1, selected, 1))?
                    .broadcast_as(rows.shape())?;
                let zeros = NativeTensor::zeros((), rows.dtype(), &context.device)?
                    .broadcast_as(rows.shape())?;
                valid.where_cond(&rows, &zeros)
            };
            keys.push(restore(input.keys, gathered_keys)?);
            values.push(restore(input.values, gathered_values)?);
            let mask = mask_row(input.mask, shape, batch, query, selected, context)?;
            let valid = valid
                .reshape((1, 1, 1, selected))?
                .broadcast_as(mask.shape())?;
            let hidden = NativeTensor::full(f32::NEG_INFINITY, (), &context.device)?
                .broadcast_as(mask.shape())?;
            masks.push(valid.where_cond(&mask, &hidden)?);
        } else if slots.iter().any(Option::is_some) {
            candle_core::bail!("missing selected history rows");
        }
        if keys.is_empty() {
            return NativeTensor::zeros(
                (1, heads, 1, input.values.dim(3) as usize),
                q.dtype(),
                &context.device,
            );
        }
        let mask = NativeTensor::cat(&masks, 3)?;
        let any = mask
            .gt(f64::NEG_INFINITY)?
            .to_dtype(DType::F32)?
            .max_keepdim(3)?
            .gt(0.)?;
        // Give fully masked rows a finite normalization input and zero
        // their resulting value. This also avoids softmax over all -inf.
        let mask = if input.sinks.is_none() {
            any.broadcast_as(mask.shape())?
                .where_cond(&mask, &NativeTensor::zeros_like(&mask)?)?
        } else {
            mask
        };
        let wrap = |native| CandleTensor::wrap(Ok(native)).map_err(candle_core::Error::wrap);
        let mask = wrap(mask)?;
        let output = CandleNeuralBackend::attention_with_sinks(
            AttentionRequest {
                queries: wrap(q.narrow(0, batch, 1)?.narrow(2, query, 1)?)?,
                keys: wrap(NativeTensor::cat(&keys, 2)?)?,
                values: wrap(NativeTensor::cat(&values, 2)?)?,
                scale: input.scale,
                arithmetic: input.arithmetic,
                mask: Some(&mask),
                sinks: input.sinks,
                softcap: None,
            },
            context,
        )
        .map_err(candle_core::Error::wrap)?
        .value;
        if input.sinks.is_some() {
            Ok(output)
        } else {
            any.broadcast_as(output.shape())?
                .where_cond(&output, &NativeTensor::zeros_like(&output)?)
        }
    })();
    if let Err(error) = context.synchronize() {
        context.quarantine((
            input.queries.clone(),
            input.mask.cloned(),
            input.sinks.cloned(),
            input.local.map(|local| {
                (
                    local.keys.clone(),
                    local.values.clone(),
                    local.mask.cloned(),
                )
            }),
            gathered,
            out.ok(),
        ));
        return Err(Error::backend_source(error));
    }
    CandleTensor::wrap(out)
}

pub(crate) fn resident_indexed_attention(
    input: &IndexedAttentionInput<'_, CandleTensor>,
    context: &CandleContext,
) -> Result<CandleTensor, Error> {
    input.validate()?;
    let mut batches = Vec::new();
    for batch in 0..input.queries.dim(0) {
        let mut rows = Vec::new();
        for query in 0..input.queries.dim(2) {
            let positions = indexed_positions(input, batch, query)?;
            let (positions, slots) = eredu_nn::compact_indexed_positions(&positions);
            let gathered = if positions.is_empty() {
                None
            } else {
                let offset = input.key_position_offset;
                let width = input.keys.dim(2);
                let indices = positions
                    .iter()
                    .map(|&position| {
                        let row = position - offset;
                        if (0..width).contains(&row) {
                            Ok(row as u32)
                        } else {
                            Err(Error::backend(
                                "indexed position is outside resident history",
                            ))
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let indices = CandleTensor::wrap(NativeTensor::new(indices, &context.device))?;
                let gather = |source: &CandleTensor| {
                    CandleTensor::wrap(
                        source
                            .value
                            .narrow(0, batch as usize, 1)
                            .and_then(|rows| rows.index_select(&indices.value, 2)),
                    )
                };
                Some([gather(input.keys)?, gather(input.values)?])
            };
            rows.push(attend_indexed_row(
                input, batch, query, &slots, gathered, context,
            )?);
        }
        batches.push(CandleTensor::concatenate(&rows, 2, context)?);
    }
    CandleTensor::concatenate(&batches, 0, context)
}
