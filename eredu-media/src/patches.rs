//! Merge-major spatial and temporal packing of already normalized host frames.

use crate::{image::NormalizedImage, MediaError};

/// Spatial and temporal patch dimensions, independent of model-family policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatiotemporalPatchSpec {
    /// Side length of a square pixel patch.
    pub patch_size: usize,
    /// Frames stored inside one flattened patch.
    pub temporal_patch_size: usize,
    /// Side length of a contiguous group of spatial patches.
    pub merge_size: usize,
}

/// Packed host patches with explicit flattened-feature and grid geometry.
#[derive(Debug, Clone)]
pub struct PackedPatches {
    values: Vec<f32>,
    shape: [usize; 2],
    grid: [usize; 3],
}

impl PackedPatches {
    /// Row-major `[patches, channels * time * patch_height * patch_width]` values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Exact flattened-patch tensor shape.
    pub const fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Temporal-group, patch-height and patch-width grid.
    pub const fn grid(&self) -> [usize; 3] {
        self.grid
    }
}

impl SpatiotemporalPatchSpec {
    /// Packs one still image or one already sampled video clip. A still image
    /// fills every temporal slot; an incomplete final group repeats its final
    /// real frame. Input frames retain their order and are never blended.
    /// Resize, normalization, frame sampling and timestamp policy belong to the
    /// caller's processor plan.
    pub fn pack(&self, frames: &[NormalizedImage]) -> Result<PackedPatches, MediaError> {
        let first = frames
            .first()
            .ok_or_else(|| MediaError::invalid("patch packing requires a frame"))?;
        let [channels, height, width] = first.shape();
        if self.patch_size == 0 || self.temporal_patch_size == 0 || self.merge_size == 0 {
            return Err(MediaError::invalid("patch dimensions must be positive"));
        }
        let factor = self
            .patch_size
            .checked_mul(self.merge_size)
            .ok_or_else(|| MediaError::invalid("patch merge factor overflow"))?;
        if height == 0
            || width == 0
            || height % factor != 0
            || width % factor != 0
            || frames.iter().any(|frame| frame.shape() != first.shape())
        {
            return Err(MediaError::invalid(
                "frames must have stable, merge-divisible spatial dimensions",
            ));
        }
        let grid_t = frames.len().div_ceil(self.temporal_patch_size);
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let patches = grid_t
            .checked_mul(grid_h)
            .and_then(|n| n.checked_mul(grid_w))
            .ok_or_else(|| MediaError::invalid("patch count overflow"))?;
        let features = channels
            .checked_mul(self.temporal_patch_size)
            .and_then(|n| n.checked_mul(self.patch_size))
            .and_then(|n| n.checked_mul(self.patch_size))
            .ok_or_else(|| MediaError::invalid("patch feature width overflow"))?;
        let count = patches
            .checked_mul(features)
            .filter(|n| *n <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| MediaError::invalid("packed patch payload overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| MediaError::invalid(format!("cannot allocate packed patches: {e}")))?;
        for time in 0..grid_t {
            for group_y in 0..grid_h / self.merge_size {
                for group_x in 0..grid_w / self.merge_size {
                    for inner_y in 0..self.merge_size {
                        for inner_x in 0..self.merge_size {
                            let y = (group_y * self.merge_size + inner_y) * self.patch_size;
                            let x = (group_x * self.merge_size + inner_x) * self.patch_size;
                            for channel in 0..channels {
                                for inner_time in 0..self.temporal_patch_size {
                                    let index = (time * self.temporal_patch_size + inner_time)
                                        .min(frames.len() - 1);
                                    for dy in 0..self.patch_size {
                                        for dx in 0..self.patch_size {
                                            let value = frames[index].get(channel, y + dy, x + dx);
                                            if !value.is_finite() {
                                                return Err(MediaError::invalid(
                                                    "packed frame contains a non-finite sample",
                                                ));
                                            }
                                            values.push(value);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(PackedPatches {
            values,
            shape: [patches, features],
            grid: [grid_t, grid_h, grid_w],
        })
    }
}
