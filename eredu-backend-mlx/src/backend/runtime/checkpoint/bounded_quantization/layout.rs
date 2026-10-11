use super::preflight::quantization_error;
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
    pub(super) writer: Arc<Mutex<PreparedCheckpointWriter>>,
    pub(super) pending_tiles: usize,
    pub(super) sealed: bool,
}

pub(super) struct BoundedAllocatorCache {
    working_set_limit_bytes: u64,
    retained_limit_bytes: u64,
    cleanup: crate::backend::ordinary_retirement::OrdinaryRetirement<AllocatorCacheCleanup>,
}

struct AllocatorCacheCleanup {
    finished: bool,
}

impl BoundedAllocatorCache {
    pub(super) fn new(working_set_limit_bytes: u64) -> Self {
        Self {
            working_set_limit_bytes,
            retained_limit_bytes: (working_set_limit_bytes / 4)
                .min(BOUNDED_QUANTIZATION_MAX_CACHE_BYTES),
            cleanup: crate::backend::ordinary_retirement::OrdinaryRetirement::new(
                AllocatorCacheCleanup { finished: false },
            ),
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
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim_all();
        memory::clear_cache()?;
        self.cleanup.finished = true;
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

impl Drop for AllocatorCacheCleanup {
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
) -> Result<Vec<OutputLayout>, Error> {
    let metadata = target.output_metadata(quantization, source_shape)?;
    target
        .output_names(quantization)?
        .into_iter()
        .map(|name| {
            let metadata = &metadata[&name];
            let dtype = match metadata.stored_dtype {
                eredu_checkpoint::StoredDtype::U32 => SafeDtype::U32,
                eredu_checkpoint::StoredDtype::U8 => SafeDtype::U8,
                eredu_checkpoint::StoredDtype::F32 => SafeDtype::F32,
                eredu_checkpoint::StoredDtype::F16 => SafeDtype::F16,
                eredu_checkpoint::StoredDtype::BF16 => SafeDtype::BF16,
                _ => return Err(quantization_error("invalid quantization output dtype")),
            };
            let rows = super::preflight::checked_product(
                &source_shape[..source_shape.len() - 1],
                "quantized rows",
            )?;
            Ok(OutputLayout {
                name,
                dtype,
                shape: metadata.physical_shape.clone(),
                row_bytes: metadata.encoded_byte_len / rows as u64,
                byte_len: metadata.encoded_byte_len,
            })
        })
        .collect()
}
