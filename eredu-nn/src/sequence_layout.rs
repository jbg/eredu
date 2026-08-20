//! Pure sequence layouts shared by patch-based encoders.

/// Failure while deriving a patch-grid sequence layout.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SequenceLayoutError {
    /// A grid must contain at least one item.
    #[error("patch grid must contain at least one item")]
    EmptyGrid,
    /// Temporal and spatial grid dimensions must be positive.
    #[error("patch grid row {index} must be positive, got {row:?}")]
    NonPositiveGrid {
        /// Row index.
        index: usize,
        /// Invalid `(time, height, width)` row.
        row: (i32, i32, i32),
    },
    /// Spatial dimensions must be divisible by the merge factor.
    #[error("patch grid row {index} spatial dimensions {height}x{width} are not divisible by merge size {merge}")]
    IndivisibleGrid {
        /// Row index.
        index: usize,
        /// Height in patches.
        height: i32,
        /// Width in patches.
        width: i32,
        /// Required merge factor.
        merge: i32,
    },
    /// Grid patch count differs from the prepared tensor.
    #[error("patch grid describes {actual} patches; expected {expected}")]
    PatchCountMismatch {
        /// Expected tensor patch count.
        expected: i32,
        /// Count derived from the grid.
        actual: i32,
    },
    /// Layout arithmetic exceeded signed 32-bit geometry.
    #[error("patch-grid layout arithmetic overflowed i32")]
    Overflow,
    /// Window geometry cannot cover one merged patch.
    #[error("window size {window_size} is too small for merge size {merge_size} and patch size {patch_size}")]
    WindowTooSmall {
        /// Window width in source coordinates.
        window_size: i32,
        /// Spatial merge factor.
        merge_size: i32,
        /// Patch width in source coordinates.
        patch_size: i32,
    },
    /// Learned position-table geometry must be positive.
    #[error("position table dimensions must be positive, got {height}x{width}")]
    NonPositiveSourceTable {
        /// Learned table height.
        height: i32,
        /// Learned table width.
        width: i32,
    },
    /// A permutation contains an out-of-range position.
    #[error("permutation entry {value} at position {position} is outside 0..{length}")]
    PermutationOutOfRange {
        /// Position containing the invalid value.
        position: usize,
        /// Invalid destination.
        value: i32,
        /// Permutation length.
        length: usize,
    },
    /// A permutation repeats a destination.
    #[error("permutation destination {value} occurs more than once")]
    DuplicatePermutation {
        /// Repeated destination index.
        value: i32,
    },
}

/// Validates positive patch grids, merge divisibility, and an optional exact patch count.
pub fn validate_patch_grid(
    grid: &[(i32, i32, i32)],
    merge_size: i32,
    expected_patches: Option<i32>,
) -> Result<i32, SequenceLayoutError> {
    if grid.is_empty() {
        return Err(SequenceLayoutError::EmptyGrid);
    }
    if merge_size <= 0 {
        return Err(SequenceLayoutError::IndivisibleGrid {
            index: 0,
            height: grid[0].1,
            width: grid[0].2,
            merge: merge_size,
        });
    }
    let mut patches = 0_i32;
    for (index, &(time, height, width)) in grid.iter().enumerate() {
        if time <= 0 || height <= 0 || width <= 0 {
            return Err(SequenceLayoutError::NonPositiveGrid {
                index,
                row: (time, height, width),
            });
        }
        if height % merge_size != 0 || width % merge_size != 0 {
            return Err(SequenceLayoutError::IndivisibleGrid {
                index,
                height,
                width,
                merge: merge_size,
            });
        }
        patches = patches
            .checked_add(
                time.checked_mul(height)
                    .and_then(|value| value.checked_mul(width))
                    .ok_or(SequenceLayoutError::Overflow)?,
            )
            .ok_or(SequenceLayoutError::Overflow)?;
    }
    if let Some(expected) = expected_patches {
        if patches != expected {
            return Err(SequenceLayoutError::PatchCountMismatch {
                expected,
                actual: patches,
            });
        }
    }
    Ok(patches)
}

/// Returns one full-attention chunk per temporal slice.
pub fn attention_chunk_lengths(grid: &[(i32, i32, i32)]) -> Result<Vec<i32>, SequenceLayoutError> {
    validate_patch_grid(grid, 1, None)?;
    let mut lengths = Vec::new();
    for &(time, height, width) in grid {
        let length = height
            .checked_mul(width)
            .ok_or(SequenceLayoutError::Overflow)?;
        lengths.extend(std::iter::repeat_n(length, time as usize));
    }
    Ok(lengths)
}

/// Window permutation over merged patch groups and its non-empty attention chunks.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WindowPartition {
    /// Source merged-group indices in window execution order.
    pub permutation: Vec<i32>,
    /// Attention token count for each non-empty window.
    pub chunk_lengths: Vec<i32>,
}

/// Coordinate transform used to sample a learned two-dimensional position table.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InterpolationMode {
    /// Maps the first and last target positions to the first and last source positions.
    AlignCorners,
    /// Uses half-pixel centers and zero contribution outside the learned table.
    HalfPixel,
}

/// Target-patch traversal used while generating interpolation samples.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PatchTraversal {
    /// Temporal slice, then row-major height and width.
    Raster,
    /// Temporal slice, spatial merge groups, then positions inside each group.
    MergeMajor(i32),
}

/// Builds `(temporal, height, width)` coordinates in the selected patch traversal.
pub fn patch_positions(
    grid: &[(i32, i32, i32)],
    traversal: PatchTraversal,
) -> Result<Vec<[i32; 3]>, SequenceLayoutError> {
    let merge = match traversal {
        PatchTraversal::Raster => 1,
        PatchTraversal::MergeMajor(merge) => merge,
    };
    validate_patch_grid(grid, merge, None)?;
    let mut positions = Vec::new();
    for &(time, height, width) in grid {
        for temporal in 0..time {
            match traversal {
                PatchTraversal::Raster => {
                    for y in 0..height {
                        for x in 0..width {
                            positions.push([temporal, y, x]);
                        }
                    }
                }
                PatchTraversal::MergeMajor(merge) => {
                    for group_y in 0..height / merge {
                        for group_x in 0..width / merge {
                            for inner_y in 0..merge {
                                for inner_x in 0..merge {
                                    positions.push([
                                        temporal,
                                        group_y * merge + inner_y,
                                        group_x * merge + inner_x,
                                    ]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(positions)
}

/// Four-corner bilinear sample for one target patch position.
#[derive(Debug, Clone, PartialEq)]
pub struct BilinearSample {
    /// Flattened row-major source-table indices in `00, 01, 10, 11` order.
    pub indices: [u32; 4],
    /// Corresponding interpolation weights.
    pub weights: [f32; 4],
}

/// Builds pure bilinear interpolation coordinates for every patch position.
pub fn bilinear_interpolation_samples(
    grid: &[(i32, i32, i32)],
    source_height: i32,
    source_width: i32,
    mode: InterpolationMode,
    traversal: PatchTraversal,
) -> Result<Vec<BilinearSample>, SequenceLayoutError> {
    if source_height <= 0 || source_width <= 0 {
        return Err(SequenceLayoutError::NonPositiveSourceTable {
            height: source_height,
            width: source_width,
        });
    }
    let merge = match traversal {
        PatchTraversal::Raster => 1,
        PatchTraversal::MergeMajor(merge) => merge,
    };
    validate_patch_grid(grid, merge, None)?;
    let mut samples = Vec::new();
    for &(time, height, width) in grid {
        for _ in 0..time {
            match traversal {
                PatchTraversal::Raster => {
                    for y in 0..height {
                        for x in 0..width {
                            samples.push(bilinear_sample(
                                y,
                                x,
                                height,
                                width,
                                source_height,
                                source_width,
                                mode,
                            )?);
                        }
                    }
                }
                PatchTraversal::MergeMajor(merge) => {
                    for group_y in 0..height / merge {
                        for group_x in 0..width / merge {
                            for inner_y in 0..merge {
                                for inner_x in 0..merge {
                                    samples.push(bilinear_sample(
                                        group_y * merge + inner_y,
                                        group_x * merge + inner_x,
                                        height,
                                        width,
                                        source_height,
                                        source_width,
                                        mode,
                                    )?);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(samples)
}

fn bilinear_sample(
    y: i32,
    x: i32,
    target_height: i32,
    target_width: i32,
    source_height: i32,
    source_width: i32,
    mode: InterpolationMode,
) -> Result<BilinearSample, SequenceLayoutError> {
    let axis = |position: i32, target: i32, source: i32| match mode {
        InterpolationMode::AlignCorners => {
            if target == 1 {
                (0, 0, 0.0)
            } else {
                let value = position as f32 * (source - 1) as f32 / (target - 1) as f32;
                let low = value.floor() as i32;
                (low, (low + 1).min(source - 1), value - low as f32)
            }
        }
        InterpolationMode::HalfPixel => {
            let value = (position as f32 + 0.5) * source as f32 / target as f32 - 0.5;
            let low = value.floor() as i32;
            (low, low + 1, value - low as f32)
        }
    };
    let (y0, y1, yf) = axis(y, target_height, source_height);
    let (x0, x1, xf) = axis(x, target_width, source_width);
    let mut indices = [0_u32; 4];
    let mut weights = [0.0_f32; 4];
    for (corner, yy, xx, weight) in [
        (0, y0, x0, (1.0 - yf) * (1.0 - xf)),
        (1, y0, x1, (1.0 - yf) * xf),
        (2, y1, x0, yf * (1.0 - xf)),
        (3, y1, x1, yf * xf),
    ] {
        let valid = yy >= 0 && yy < source_height && xx >= 0 && xx < source_width;
        let yy = yy.clamp(0, source_height - 1);
        let xx = xx.clamp(0, source_width - 1);
        let index = yy
            .checked_mul(source_width)
            .and_then(|value| value.checked_add(xx))
            .ok_or(SequenceLayoutError::Overflow)?;
        indices[corner] = u32::try_from(index).map_err(|_| SequenceLayoutError::Overflow)?;
        weights[corner] = if valid { weight } else { 0.0 };
    }
    Ok(BilinearSample { indices, weights })
}

/// Partitions merged spatial patch groups into windows without materializing tensors.
pub fn window_partition(
    grid: &[(i32, i32, i32)],
    merge_size: i32,
    window_size: i32,
    patch_size: i32,
) -> Result<WindowPartition, SequenceLayoutError> {
    validate_patch_grid(grid, merge_size, None)?;
    if patch_size <= 0 || window_size <= 0 {
        return Err(SequenceLayoutError::WindowTooSmall {
            window_size,
            merge_size,
            patch_size,
        });
    }
    let merged_window = window_size / merge_size / patch_size;
    if merged_window <= 0 {
        return Err(SequenceLayoutError::WindowTooSmall {
            window_size,
            merge_size,
            patch_size,
        });
    }
    let merge_unit = merge_size
        .checked_mul(merge_size)
        .ok_or(SequenceLayoutError::Overflow)?;
    let mut permutation = Vec::new();
    let mut chunk_lengths = Vec::new();
    let mut item_offset = 0_i32;
    for &(time, height, width) in grid {
        let merged_height = height / merge_size;
        let merged_width = width / merge_size;
        let windows_height = div_ceil(merged_height, merged_window)?;
        let windows_width = div_ceil(merged_width, merged_window)?;
        for temporal in 0..time {
            for window_y in 0..windows_height {
                for window_x in 0..windows_width {
                    let mut groups = 0_i32;
                    for inner_y in 0..merged_window {
                        for inner_x in 0..merged_window {
                            let y = window_y * merged_window + inner_y;
                            let x = window_x * merged_window + inner_x;
                            if y < merged_height && x < merged_width {
                                let index = temporal
                                    .checked_mul(merged_height)
                                    .and_then(|value| value.checked_mul(merged_width))
                                    .and_then(|value| {
                                        y.checked_mul(merged_width)
                                            .and_then(|row| value.checked_add(row))
                                    })
                                    .and_then(|value| value.checked_add(x))
                                    .and_then(|value| value.checked_add(item_offset))
                                    .ok_or(SequenceLayoutError::Overflow)?;
                                permutation.push(index);
                                groups += 1;
                            }
                        }
                    }
                    if groups > 0 {
                        chunk_lengths.push(
                            groups
                                .checked_mul(merge_unit)
                                .ok_or(SequenceLayoutError::Overflow)?,
                        );
                    }
                }
            }
        }
        item_offset = item_offset
            .checked_add(
                time.checked_mul(merged_height)
                    .and_then(|value| value.checked_mul(merged_width))
                    .ok_or(SequenceLayoutError::Overflow)?,
            )
            .ok_or(SequenceLayoutError::Overflow)?;
    }
    Ok(WindowPartition {
        permutation,
        chunk_lengths,
    })
}

/// Validates a permutation and returns its inverse.
pub fn inverse_permutation(indices: &[i32]) -> Result<Vec<i32>, SequenceLayoutError> {
    let mut inverse = vec![-1; indices.len()];
    for (position, &index) in indices.iter().enumerate() {
        let destination =
            usize::try_from(index).map_err(|_| SequenceLayoutError::PermutationOutOfRange {
                position,
                value: index,
                length: indices.len(),
            })?;
        let slot =
            inverse
                .get_mut(destination)
                .ok_or(SequenceLayoutError::PermutationOutOfRange {
                    position,
                    value: index,
                    length: indices.len(),
                })?;
        if *slot != -1 {
            return Err(SequenceLayoutError::DuplicatePermutation { value: index });
        }
        *slot = position as i32;
    }
    Ok(inverse)
}

fn div_ceil(value: i32, divisor: i32) -> Result<i32, SequenceLayoutError> {
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or(SequenceLayoutError::Overflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_grid_count_and_merge_geometry() {
        assert_eq!(validate_patch_grid(&[(2, 4, 6)], 2, Some(48)), Ok(48));
        assert!(matches!(
            validate_patch_grid(&[(1, 3, 4)], 2, None),
            Err(SequenceLayoutError::IndivisibleGrid { .. })
        ));
        assert!(matches!(
            validate_patch_grid(&[(1, 4, 4)], 2, Some(15)),
            Err(SequenceLayoutError::PatchCountMismatch { .. })
        ));
    }

    #[test]
    fn window_partition_matches_merged_group_order_and_chunks() {
        let layout = window_partition(&[(1, 4, 6)], 2, 4, 1).unwrap();
        assert_eq!(layout.permutation, vec![0, 1, 3, 4, 2, 5]);
        assert_eq!(layout.chunk_lengths, vec![16, 8]);
        assert_eq!(
            inverse_permutation(&layout.permutation).unwrap(),
            vec![0, 1, 4, 2, 3, 5]
        );
    }

    #[test]
    fn multiple_items_use_disjoint_offsets_and_temporal_chunks() {
        let layout = window_partition(&[(2, 2, 2), (1, 2, 4)], 1, 2, 1).unwrap();
        assert_eq!(
            layout.permutation,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        );
        assert_eq!(layout.chunk_lengths, vec![4, 4, 4, 4]);
        assert_eq!(attention_chunk_lengths(&[(2, 2, 3)]).unwrap(), vec![6, 6]);
    }

    #[test]
    fn patch_positions_follow_raster_and_merge_major_traversal() {
        let grid = [(1, 2, 2)];
        assert_eq!(
            patch_positions(&grid, PatchTraversal::Raster).unwrap(),
            vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
        );
        assert_eq!(
            patch_positions(&grid, PatchTraversal::MergeMajor(2)).unwrap(),
            vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
        );
    }

    #[test]
    fn malformed_permutations_fail_instead_of_indexing_out_of_bounds() {
        assert!(matches!(
            inverse_permutation(&[0, 0]),
            Err(SequenceLayoutError::DuplicatePermutation { .. })
        ));
        assert!(matches!(
            inverse_permutation(&[0, 2]),
            Err(SequenceLayoutError::PermutationOutOfRange { .. })
        ));
    }

    #[test]
    fn interpolation_modes_preserve_general_coordinate_semantics() {
        let aligned = bilinear_interpolation_samples(
            &[(1, 2, 2)],
            3,
            3,
            InterpolationMode::AlignCorners,
            PatchTraversal::MergeMajor(1),
        )
        .unwrap();
        assert_eq!(aligned[0].indices, [0, 1, 3, 4]);
        assert_eq!(aligned[3].indices, [8, 8, 8, 8]);
        assert_eq!(aligned[3].weights, [1.0, 0.0, 0.0, 0.0]);

        let half_pixel = bilinear_interpolation_samples(
            &[(1, 2, 2)],
            1,
            1,
            InterpolationMode::HalfPixel,
            PatchTraversal::Raster,
        )
        .unwrap();
        assert_eq!(half_pixel[0].indices, [0, 0, 0, 0]);
        assert!(half_pixel[0].weights.iter().sum::<f32>() < 1.0);
    }
}
