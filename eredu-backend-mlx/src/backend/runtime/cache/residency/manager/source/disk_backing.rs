//! Exact authenticated file metadata; no file open or execution authority.
use super::*;

impl CacheBlockSource<'_> {
    /// Validates the retained Live writer or authenticated persistent import.
    /// Path labels and legacy buffered files do not supply either source.
    pub(crate) fn retained_disk_types(self) -> Result<Option<[Dtype; 2]>, CacheSourceError> {
        let Some(disk) = self.disk() else {
            return Ok(None);
        };
        let source = disk
            .file_source()
            .ok_or(CacheSourceError::PromotionRequired)?;
        let layout = source.layout().ok_or(CacheSourceError::PromotionRequired)?;
        if disk.buffered().is_some()
            || !source.owns_declared_backing()
            || disk.path() != source.path()
            || disk.names() != layout.names()
            || source.file_bytes() != Some(layout.file_bytes())
        {
            return Err(CacheSourceError::Identity);
        }
        let metadata = layout.tensor_metadata();
        let mut types = [Dtype::Float32; 2];
        let mut total = 0u64;
        for index in 0..2 {
            let (shape, stored, bytes) = metadata[index];
            let (native, name) = match stored {
                safetensors::tensor::Dtype::F16 => (Dtype::Float16, "Float16"),
                safetensors::tensor::Dtype::BF16 => (Dtype::Bfloat16, "Bfloat16"),
                safetensors::tensor::Dtype::F32 => (Dtype::Float32, "Float32"),
                _ => return Err(CacheSourceError::Geometry),
            };
            if shape.len() != self.shapes()[index].len()
                || shape
                    .iter()
                    .zip(self.shapes()[index])
                    .any(|(a, b)| usize::try_from(*b).ok() != Some(*a))
                || self.dtypes()[index] != name
            {
                return Err(CacheSourceError::Geometry);
            }
            types[index] = native;
            let logical = shape
                .iter()
                .try_fold(
                    CacheBlockMetadata::floating_dtype_bytes(native)
                        .ok_or(CacheSourceError::Geometry)?,
                    |sum, dim| sum.checked_mul(u64::try_from(*dim).ok()?),
                )
                .ok_or(CacheSourceError::Overflow)?;
            if logical != u64::try_from(bytes).map_err(|_| CacheSourceError::Overflow)? {
                return Err(CacheSourceError::Geometry);
            }
            total = total
                .checked_add(logical)
                .ok_or(CacheSourceError::Overflow)?;
        }
        if total != self.logical_bytes()
            || CacheBlockMetadata::floating_bytes(self.shapes(), types)? != total
        {
            return Err(CacheSourceError::Geometry);
        }
        Ok(Some(types))
    }
    pub(crate) fn retained_disk_control_bytes() -> usize {
        size_of::<(
            Self,
            Option<CacheDiskSource<'static>>,
            Option<CacheFileSource>,
            Option<&'static eredu_runtime::cache::CacheShardLayout>,
            [(&'static [usize], safetensors::tensor::Dtype, usize); 2],
            [Dtype; 2],
            (usize, Dtype, &'static str, u64),
            Result<Option<[Dtype; 2]>, CacheSourceError>,
            std::iter::Zip<std::slice::Iter<'static, usize>, std::slice::Iter<'static, i32>>,
            std::slice::Iter<'static, usize>,
        )>()
    }
}
