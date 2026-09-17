//! Checked borrowed-grid kernels shared by ordinary and prepared storage.

use super::{
    bilinear_sample, div_ceil, BilinearSample, InterpolationMode, PatchTraversal,
    SequenceLayoutError,
};

/// Exact scalar population of a validated patch-grid sequence.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PatchGridPopulation {
    /// Flattened patch count, already checked against native signed geometry.
    pub patches: i32,
    /// Number of nonempty temporal slices.
    pub frames: usize,
    /// Number of grid rows.
    pub rows: usize,
}

/// Validates borrowed rows without constructing a flattened grid collection.
pub fn patch_grid_population(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    merge: i32,
    expected: Option<i32>,
) -> Result<PatchGridPopulation, SequenceLayoutError> {
    let first = rows.clone().next().ok_or(SequenceLayoutError::EmptyGrid)?;
    if merge <= 0 {
        return Err(SequenceLayoutError::IndivisibleGrid {
            index: 0,
            height: first.1,
            width: first.2,
            merge,
        });
    }
    let mut population = PatchGridPopulation {
        patches: 0,
        frames: 0,
        rows: 0,
    };
    for (index, row @ (time, height, width)) in rows.enumerate() {
        if time <= 0 || height <= 0 || width <= 0 {
            return Err(SequenceLayoutError::NonPositiveGrid { index, row });
        }
        if height % merge != 0 || width % merge != 0 {
            return Err(SequenceLayoutError::IndivisibleGrid {
                index,
                height,
                width,
                merge,
            });
        }
        let patches = time
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or(SequenceLayoutError::Overflow)?;
        population.patches = population
            .patches
            .checked_add(patches)
            .ok_or(SequenceLayoutError::Overflow)?;
        population.frames = population
            .frames
            .checked_add(usize::try_from(time).map_err(|_| SequenceLayoutError::Overflow)?)
            .ok_or(SequenceLayoutError::Overflow)?;
        population.rows = population
            .rows
            .checked_add(1)
            .ok_or(SequenceLayoutError::Overflow)?;
    }
    if let Some(expected) = expected {
        if expected != population.patches {
            return Err(SequenceLayoutError::PatchCountMismatch {
                expected,
                actual: population.patches,
            });
        }
    }
    Ok(population)
}

fn require_output(actual: usize, required: usize) -> Result<(), SequenceLayoutError> {
    if actual < required {
        return Err(SequenceLayoutError::DestinationTooShort { required, actual });
    }
    Ok(())
}

/// Validates every direct raster lookup, including the greatest actual index.
pub fn gathered_position_population(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
) -> Result<PatchGridPopulation, SequenceLayoutError> {
    if source_height <= 0 || source_width <= 0 {
        return Err(SequenceLayoutError::NonPositiveSourceTable {
            height: source_height,
            width: source_width,
        });
    }
    let population = patch_grid_population(rows.clone(), 1, None)?;
    for (index, (_, height, width)) in rows.enumerate() {
        if height > source_height || width > source_width {
            return Err(SequenceLayoutError::GridExceedsSourceTable {
                index,
                height,
                width,
                source_height,
                source_width,
            });
        }
        (height - 1)
            .checked_mul(source_width)
            .and_then(|value| value.checked_add(width - 1))
            .ok_or(SequenceLayoutError::Overflow)?;
    }
    Ok(population)
}

/// Visits direct learned-position indices in temporal then raster order.
pub fn visit_gathered_patch_positions(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
    mut emit: impl FnMut(i32),
) -> Result<PatchGridPopulation, SequenceLayoutError> {
    let population = gathered_position_population(rows.clone(), source_height, source_width)?;
    visit_patch_positions(rows, PatchTraversal::Raster, |[_, y, x]| {
        emit(y * source_width + x);
    })?;
    Ok(population)
}

/// Fills direct learned-position indices only after complete source/capacity checks.
pub fn fill_gathered_patch_positions(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
    output: &mut [i32],
) -> Result<usize, SequenceLayoutError> {
    let population = gathered_position_population(rows.clone(), source_height, source_width)?;
    let required =
        usize::try_from(population.patches).map_err(|_| SequenceLayoutError::Overflow)?;
    require_output(output.len(), required)?;
    let mut at = 0;
    visit_gathered_patch_positions(rows, source_height, source_width, |index| {
        output[at] = index;
        at += 1;
    })?;
    Ok(at)
}

/// Writes one length per temporal slice after validating the entire population.
/// An error leaves the destination unchanged; unused tail cells are untouched.
pub fn fill_attention_chunk_lengths(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    output: &mut [i32],
) -> Result<usize, SequenceLayoutError> {
    let population = patch_grid_population(rows.clone(), 1, None)?;
    require_output(output.len(), population.frames)?;
    let mut at = 0;
    for (time, height, width) in rows {
        // The complete grid product was checked before any destination write.
        let length = height * width;
        for _ in 0..time {
            output[at] = length;
            at += 1;
        }
    }
    Ok(at)
}

/// Visits the existing patch traversal without an intermediate coordinate Vec.
/// All grid validation precedes the first callback. The callback owns any work
/// it chooses to perform; this pure kernel grants no allocation authority.
pub fn visit_patch_positions(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    traversal: PatchTraversal,
    mut emit: impl FnMut([i32; 3]),
) -> Result<PatchGridPopulation, SequenceLayoutError> {
    let merge = match traversal {
        PatchTraversal::Raster => 1,
        PatchTraversal::MergeMajor(merge) => merge,
    };
    let population = patch_grid_population(rows.clone(), merge, None)?;
    for (time, height, width) in rows {
        for temporal in 0..time {
            match traversal {
                PatchTraversal::Raster => {
                    for y in 0..height {
                        for x in 0..width {
                            emit([temporal, y, x]);
                        }
                    }
                }
                PatchTraversal::MergeMajor(merge) => {
                    for group_y in 0..height / merge {
                        for group_x in 0..width / merge {
                            for inner_y in 0..merge {
                                for inner_x in 0..merge {
                                    emit([
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
    Ok(population)
}

/// Writes only the y/x coordinates required by the two-axis vision equation.
pub fn fill_spatial_patch_positions(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    traversal: PatchTraversal,
    output: &mut [i32],
) -> Result<usize, SequenceLayoutError> {
    let merge = match traversal {
        PatchTraversal::Raster => 1,
        PatchTraversal::MergeMajor(merge) => merge,
    };
    let population = patch_grid_population(rows.clone(), merge, None)?;
    let cells = usize::try_from(population.patches)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .ok_or(SequenceLayoutError::Overflow)?;
    require_output(output.len(), cells)?;
    let mut at = 0;
    visit_patch_positions(rows, traversal, |[_, y, x]| {
        output[at] = y;
        output[at + 1] = x;
        at += 2;
    })?;
    Ok(at)
}

/// Checks interpolation geometry before any destination or callback is used.
pub fn interpolation_population(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
    mode: InterpolationMode,
    traversal: PatchTraversal,
) -> Result<PatchGridPopulation, SequenceLayoutError> {
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
    let population = patch_grid_population(rows.clone(), merge, None)?;
    // Coordinates and flattened corner indices are monotone in each target
    // axis. Check the greatest accessed corner for every item before writing.
    // Do not reject an unused overflowing source-table extent: a one-position
    // AlignCorners target can legally address only the source's first cell.
    for (_, height, width) in rows {
        bilinear_sample(
            height - 1,
            width - 1,
            height,
            width,
            source_height,
            source_width,
            mode,
        )?;
    }
    Ok(population)
}

/// Visits unchanged four-corner sample equations in the selected traversal.
pub fn visit_bilinear_samples(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
    mode: InterpolationMode,
    traversal: PatchTraversal,
    mut emit: impl FnMut(BilinearSample),
) -> Result<PatchGridPopulation, SequenceLayoutError> {
    let population =
        interpolation_population(rows.clone(), source_height, source_width, mode, traversal)?;
    for (time, height, width) in rows {
        // Reuse the exact traversal for each item; global and per-item
        // validation use the same checked geometry and perform no allocation.
        let mut failure = None;
        visit_patch_positions(
            std::iter::once((time, height, width)),
            traversal,
            |[_, y, x]| {
                if failure.is_none() {
                    match bilinear_sample(y, x, height, width, source_height, source_width, mode) {
                        Ok(sample) => emit(sample),
                        Err(error) => failure = Some(error),
                    }
                }
            },
        )?;
        if let Some(error) = failure {
            return Err(error);
        }
    }
    Ok(population)
}

/// Writes corner-major native-index and weight planes without sample storage.
/// Their required lengths are both four times the checked patch population.
pub fn fill_bilinear_planes(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    source_height: i32,
    source_width: i32,
    mode: InterpolationMode,
    traversal: PatchTraversal,
    indices: &mut [i32],
    weights: &mut [f32],
) -> Result<usize, SequenceLayoutError> {
    let population =
        interpolation_population(rows.clone(), source_height, source_width, mode, traversal)?;
    let patches = usize::try_from(population.patches).map_err(|_| SequenceLayoutError::Overflow)?;
    let cells = patches
        .checked_mul(4)
        .ok_or(SequenceLayoutError::Overflow)?;
    require_output(indices.len(), cells)?;
    require_output(weights.len(), cells)?;
    let mut position = 0;
    visit_bilinear_samples(
        rows,
        source_height,
        source_width,
        mode,
        traversal,
        |sample| {
            for corner in 0..4 {
                // Preflight checked the greatest accessed signed source index.
                indices[corner * patches + position] = sample.indices[corner] as i32;
                weights[corner * patches + position] = sample.weights[corner];
            }
            position += 1;
        },
    )?;
    Ok(patches)
}

/// Exact window-table population; no padded window tensor or host table exists.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WindowPartitionPopulation {
    /// Number of merged patch groups, and inverse-permutation cells.
    pub groups: usize,
    /// Number of nonempty attention windows.
    pub windows: usize,
}

/// Counts the same nonempty windows used by the execution-order permutation.
pub fn window_partition_population(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    merge: i32,
    window_size: i32,
    patch_size: i32,
) -> Result<WindowPartitionPopulation, SequenceLayoutError> {
    let population = patch_grid_population(rows.clone(), merge, None)?;
    if patch_size <= 0 || window_size <= 0 {
        return Err(SequenceLayoutError::WindowTooSmall {
            window_size,
            merge_size: merge,
            patch_size,
        });
    }
    let window = window_size / merge / patch_size;
    if window <= 0 {
        return Err(SequenceLayoutError::WindowTooSmall {
            window_size,
            merge_size: merge,
            patch_size,
        });
    }
    let unit = merge
        .checked_mul(merge)
        .ok_or(SequenceLayoutError::Overflow)?;
    let mut windows = 0usize;
    for (time, height, width) in rows {
        let count = time
            .checked_mul(div_ceil(height / merge, window)?)
            .and_then(|n| n.checked_mul(div_ceil(width / merge, window).ok()?))
            .ok_or(SequenceLayoutError::Overflow)?;
        windows = windows
            .checked_add(usize::try_from(count).map_err(|_| SequenceLayoutError::Overflow)?)
            .ok_or(SequenceLayoutError::Overflow)?;
    }
    Ok(WindowPartitionPopulation {
        groups: usize::try_from(population.patches / unit)
            .map_err(|_| SequenceLayoutError::Overflow)?,
        windows,
    })
}

/// Visits the bijective window order and lengths after complete scalar checks.
pub fn visit_window_partition(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    merge: i32,
    window_size: i32,
    patch_size: i32,
    mut emit_index: impl FnMut(i32),
    mut emit_chunk: impl FnMut(i32),
) -> Result<WindowPartitionPopulation, SequenceLayoutError> {
    let population = window_partition_population(rows.clone(), merge, window_size, patch_size)?;
    let window = window_size / merge / patch_size;
    let unit = merge * merge;
    let (mut at, mut item_offset) = (0usize, 0i32);
    for (time, height, width) in rows {
        let (height, width) = (height / merge, width / merge);
        // These exact calls succeeded in the population preflight.
        let (windows_y, windows_x) = (div_ceil(height, window)?, div_ceil(width, window)?);
        for temporal in 0..time {
            for wy in 0..windows_y {
                for wx in 0..windows_x {
                    let begin = at;
                    // Clip each window instead of iterating its padded area.
                    // i64 avoids an overflow in the final padded window end.
                    let y0 = wy * window;
                    let x0 = wx * window;
                    let y1 = (i64::from(y0) + i64::from(window)).min(i64::from(height)) as i32;
                    let x1 = (i64::from(x0) + i64::from(window)).min(i64::from(width)) as i32;
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let index = item_offset + temporal * height * width + y * width + x;
                            emit_index(index);
                            at += 1;
                        }
                    }
                    emit_chunk((at - begin) as i32 * unit);
                }
            }
        }
        item_offset += time * height * width;
    }
    Ok(population)
}

/// Fills the permutation, its inverse and nonempty chunks after full preflight.
/// The inverse is produced from the same bijective traversal, not an allocating
/// duplicate-validation pass over untrusted caller-provided indices.
pub fn fill_window_partition(
    rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
    merge: i32,
    window_size: i32,
    patch_size: i32,
    permutation: &mut [i32],
    inverse: &mut [i32],
    chunks: &mut [i32],
) -> Result<WindowPartitionPopulation, SequenceLayoutError> {
    let population = window_partition_population(rows.clone(), merge, window_size, patch_size)?;
    require_output(permutation.len(), population.groups)?;
    require_output(inverse.len(), population.groups)?;
    require_output(chunks.len(), population.windows)?;
    let (mut at, mut chunk_at) = (0usize, 0usize);
    visit_window_partition(
        rows,
        merge,
        window_size,
        patch_size,
        |index| {
            permutation[at] = index;
            inverse[index as usize] = at as i32;
            at += 1;
        },
        |length| {
            chunks[chunk_at] = length;
            chunk_at += 1;
        },
    )?;
    Ok(population)
}

#[cfg(test)]
#[path = "fixed_tests.rs"]
mod tests;
