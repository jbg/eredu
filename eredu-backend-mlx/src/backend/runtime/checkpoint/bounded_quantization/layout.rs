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
    pub(super) layouts: Vec<OutputLayout>,
    pub(super) buffers: Vec<Vec<u8>>,
    pub(super) pending_tiles: usize,
    pub(super) sealed: bool,
}

pub(super) struct BoundedAllocatorCache {
    working_set_limit_bytes: u64,
    retained_limit_bytes: u64,
    finished: bool,
}

impl BoundedAllocatorCache {
    pub(super) fn new(working_set_limit_bytes: u64) -> Self {
        Self {
            working_set_limit_bytes,
            retained_limit_bytes: (working_set_limit_bytes / 4)
                .min(BOUNDED_QUANTIZATION_MAX_CACHE_BYTES),
            finished: false,
        }
    }

    pub(super) fn begin(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        Ok(())
    }

    pub(super) fn prepare_submission(
        &mut self,
        queued_working_set_bytes: u64,
        incoming_tile_bytes: u64,
    ) -> Result<(), Error> {
        let planned_bytes = queued_working_set_bytes
            .checked_add(incoming_tile_bytes)
            .ok_or_else(|| quantization_error("conversion submission working-set overflow"))?;
        if planned_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "conversion submission requires {planned_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(planned_bytes)
    }

    pub(super) fn tile_completed(&mut self, queued_working_set_bytes: u64) -> Result<(), Error> {
        if queued_working_set_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "queued conversion tiles require {queued_working_set_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(queued_working_set_bytes)
    }

    fn clear_if_needed(&mut self, active_working_set_bytes: u64) -> Result<(), Error> {
        let cached_bytes = u64::try_from(memory::cache_memory()?)
            .map_err(|_| quantization_error("allocator-cache bytes are not representable"))?;
        let available_cache_bytes = self
            .working_set_limit_bytes
            .checked_sub(active_working_set_bytes)
            .expect("validated active conversion working set");
        if allocator_cache_requires_clear(
            cached_bytes,
            self.retained_limit_bytes,
            available_cache_bytes,
        ) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        self.finished = true;
        Ok(())
    }
}

pub(super) const fn allocator_cache_requires_clear(
    cached_bytes: u64,
    retained_limit_bytes: u64,
    available_cache_bytes: u64,
) -> bool {
    cached_bytes > retained_limit_bytes || cached_bytes > available_cache_bytes
}

impl Drop for BoundedAllocatorCache {
    fn drop(&mut self) {
        if !self.finished {
            let _ = memory::clear_cache();
        }
    }
}

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
            )))
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

pub(super) fn allocate_output_buffer(name: &str, byte_len: u64) -> Result<Vec<u8>, Error> {
    let byte_len = usize::try_from(byte_len)
        .map_err(|_| quantization_error(format!("output {name:?} is too large for memory")))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(byte_len).map_err(|error| {
        quantization_error(format!(
            "cannot allocate {byte_len} bytes for in-memory quantized output {name:?}: {error}"
        ))
    })?;
    bytes.resize(byte_len, 0);
    Ok(bytes)
}
