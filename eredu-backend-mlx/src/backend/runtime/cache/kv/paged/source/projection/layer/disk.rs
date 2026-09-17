//! Exact own-writer file lifetime retained beside the canonical source pins.
use super::*;
use eredu_runtime::cache::LiveCacheBlockSource;

pub(super) struct RetainedPagedDisk {
    id: CacheBlockId,
    source: LiveCacheBlockSource,
}
impl RetainedPagedDisk {
    pub(super) fn prepare(
        block: CacheBlockSource<'_>,
        key_only: bool,
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(Self::controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let disk = block
            .disk()
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        let source = disk
            .live_file()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        let layout = source
            .writer_layout()
            .ok_or_else(|| fail(CacheSourceError::PromotionRequired))?;
        if disk.persistent()
            || disk.buffered().is_some()
            || disk.path() != source.path()
            || disk.names() != layout.names()
            || source.file_bytes() != Some(layout.file_bytes())
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let pair = block_geometry(block, key_only).map_err(fail)?;
        let metadata = layout.tensor_metadata();
        for index in 0..2 {
            let (shape, dtype, bytes) = metadata[index];
            let native = match dtype {
                safetensors::tensor::Dtype::F16 => Dtype::Float16,
                safetensors::tensor::Dtype::BF16 => Dtype::Bfloat16,
                safetensors::tensor::Dtype::F32 => Dtype::Float32,
                _ => return Err(fail(CacheSourceError::Geometry)),
            };
            if shape.len() != 4
                || shape
                    .iter()
                    .zip(pair[index].shape)
                    .any(|(a, b)| usize::try_from(b).ok() != Some(*a))
                || native != pair[index].dtype
                || u64::try_from(bytes).ok() != Some(pair[index].logical_bytes)
            {
                return Err(fail(CacheSourceError::Geometry));
            }
        }
        Ok(Self {
            id: block.id().clone(),
            source: source.clone(),
        })
    }
    pub(super) fn controls() -> Option<usize> {
        Some(size_of::<(
            Self,
            CacheBlockSource<'_>,
            bool,
            &WorkspaceContext,
            Result<Self, CacheSourceFailure>,
            [PagedCacheArrayGeometry; 2],
            [(&[usize], safetensors::tensor::Dtype, usize); 2],
            (usize, Dtype),
            Option<&LiveCacheBlockSource>,
        )>())
    }
}
impl ProjectedPagedSource {
    pub(crate) fn retained_file(&self, id: &CacheBlockId) -> Option<&LiveCacheBlockSource> {
        self.files
            .iter()
            .find(|entry| &entry.id == id)
            .map(|entry| &entry.source)
    }
    pub(in super::super) fn validate_files(
        &self,
        loan: &CacheBlockSourceLoan<'_>,
    ) -> Result<(), CacheSourceError> {
        for file in &self.files {
            let row = loan
                .blocks()
                .find(|row| row.id() == &file.id)
                .ok_or(CacheSourceError::Identity)?;
            if row
                .disk()
                .and_then(|disk| disk.live_file())
                .is_none_or(|actual| !file.source.same_source(actual))
            {
                return Err(CacheSourceError::Identity);
            }
        }
        Ok(())
    }
}
