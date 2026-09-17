//! Unchanged pre-factor equations used only as an independent bounded oracle.
//! Exact movement origins are recorded in the B3 stage ledger.

use super::{BilinearSample, InterpolationMode, PatchTraversal, SequenceLayoutError, WindowPartition};

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

fn div_ceil(value: i32, divisor: i32) -> Result<i32, SequenceLayoutError> {
    value
        .checked_add(divisor - 1)
        .map(|value| value / divisor)
        .ok_or(SequenceLayoutError::Overflow)
}
