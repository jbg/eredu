use super::preflight::{checked_product, quantization_error};
use super::*;

#[derive(Debug, Clone)]
pub(super) struct OutputLayout {
    pub(super) name: String,
    pub(super) dtype: SafeDtype,
    pub(super) shape: Vec<usize>,
    pub(super) row_bytes: u64,
    pub(super) byte_len: u64,
}

pub(super) struct OutputShard {
    pub(super) geometry: super::preparation::ConversionGeometry,
    pub(super) layouts: Vec<OutputLayout>,
    pub(super) buffers: Vec<eredu_checkpoint::store::MemoryTensorBuffer>,
    pub(super) pending_tiles: usize,
    pub(super) sealed: bool,
}

mod cache;
pub(super) use cache::{allocator_cache_requires_clear, BoundedAllocatorCache, CacheAdmissionError};

impl OutputShard {
    pub(super) fn tile_submitted(&mut self) {
        debug_assert!(!self.sealed);
        self.pending_tiles = self
            .pending_tiles
            .checked_add(1)
            .expect("output shard pending-tile count overflowed");
    }

    pub(super) fn tile_completed(&mut self) -> Result<(), Error> {
        self.pending_tiles = self
            .pending_tiles
            .checked_sub(1)
            .expect("completed output tile was previously submitted");
        self.close_if_complete()
    }

    pub(super) fn seal(&mut self) -> Result<(), Error> {
        self.sealed = true;
        self.close_if_complete()
    }

    fn close_if_complete(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

pub(super) fn output_layouts(
    target: &BoundedQuantizationTarget,
    quantization: WeightQuantization,
    source_shape: &[usize],
    rows: usize,
    columns: usize,
    bits: usize,
) -> Result<Vec<OutputLayout>, Error> {
    let scales_name = target.scales_name.clone();
    let biases_name = target.biases_name.clone();
    let packed_columns = columns
        .checked_mul(bits)
        .and_then(|bits| bits.checked_div(32))
        .ok_or_else(|| quantization_error("packed column count overflow"))?;
    let scale_columns = columns
        .checked_div(quantization.group_size() as usize)
        .ok_or_else(|| quantization_error("scale column count overflow"))?;
    let mut prefix = source_shape[..source_shape.len() - 2].to_vec();
    prefix.push(rows);

    let mut layouts = Vec::with_capacity(if quantization.has_biases() { 3 } else { 2 });
    let (affine_dtype, affine_scalar_bytes) = match target.affine_companion_dtype {
        RecipeDtype::F16 => (SafeDtype::F16, 2),
        RecipeDtype::BF16 => (SafeDtype::BF16, 2),
        RecipeDtype::F32 => (SafeDtype::F32, 4),
        _ => {
            return Err(quantization_error(format!(
                "bounded affine companion output has invalid dtype {:?}",
                target.affine_companion_dtype
            )));
        }
    };
    layouts.push(layout(
        target.weight_name.clone(),
        SafeDtype::U32,
        prefix.clone(),
        packed_columns,
        4,
    )?);
    layouts.push(layout(
        scales_name,
        if matches!(quantization, WeightQuantization::MxFp4) {
            SafeDtype::U8
        } else {
            affine_dtype
        },
        prefix.clone(),
        scale_columns,
        if matches!(quantization, WeightQuantization::MxFp4) {
            1
        } else {
            affine_scalar_bytes
        },
    )?);
    if quantization.has_biases() {
        layouts.push(layout(
            biases_name.expect("bounded plan validated affine-bias identity"),
            affine_dtype,
            prefix,
            scale_columns,
            affine_scalar_bytes,
        )?);
    }

    Ok(layouts)
}

fn layout(
    name: String,
    dtype: SafeDtype,
    mut prefix: Vec<usize>,
    columns: usize,
    scalar_bytes: u64,
) -> Result<OutputLayout, Error> {
    prefix.push(columns);
    let row_bytes = (columns as u64)
        .checked_mul(scalar_bytes)
        .ok_or_else(|| quantization_error("quantized row size overflow"))?;
    let rows = checked_product(&prefix[..prefix.len() - 1], "quantized output rows")?;
    let byte_len = (rows as u64)
        .checked_mul(row_bytes)
        .ok_or_else(|| quantization_error("quantized tensor size overflow"))?;
    Ok(OutputLayout {
        name,
        dtype,
        shape: prefix,
        row_bytes,
        byte_len,
    })
}
