//! A checked fixed table layout for patch encoders, independent of model families.
use super::*;
use crate::multimodal::{
    MultiAxisRotaryLayout, MultiAxisRotarySpecRef, PreparedMultiAxisRotary, RotaryAxisSpec,
    RotaryTableError,
};

/// Learned spatial-position lookup equation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PatchPositionTableSpec {
    /// Direct raster lookup into a learned rectangular table.
    Gathered { height: i32, width: i32 },
    /// Four-corner interpolation using the declared coordinate/traversal policy.
    Interpolated {
        height: i32,
        width: i32,
        mode: InterpolationMode,
        traversal: PatchTraversal,
    },
}
/// Attention execution order for merged patch groups.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PatchAttentionWindows {
    /// Identity merged order and complete temporal-slice attention.
    Full,
    /// Shared nonempty spatial window equation.
    Windowed { window_size: i32, patch_size: i32 },
}
/// Complete source-table equations selected by an architecture.
/// This is numerical geometry, not a source identity or allocation grant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchEncoderTableSpec {
    /// Merge factor for encoder execution order.
    pub merge: i32,
    /// Learned spatial position policy.
    pub positions: PatchPositionTableSpec,
    /// Attention permutation/chunk policy.
    pub windows: PatchAttentionWindows,
    /// Height/width rotary feature policies.
    pub rotary_axes: [RotaryAxisSpec; 2],
    /// Base wavelength.
    pub rotary_base: f32,
    /// Minimum offset position.
    pub rotary_minimum: i32,
    /// Rotary feature arrangement.
    pub rotary_layout: MultiAxisRotaryLayout,
}
impl PatchEncoderTableSpec {
    fn rotary(&self) -> MultiAxisRotarySpecRef<'_> {
        MultiAxisRotarySpecRef {
            axes: &self.rotary_axes,
            base: self.rotary_base,
            minimum_position: self.rotary_minimum,
            layout: self.rotary_layout,
        }
    }
}

/// Fixed preflight/fill failure; no formatted error is constructed here.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PatchEncoderTableError {
    /// Patch, window, interpolation or destination failure.
    #[error(transparent)]
    Sequence(#[from] SequenceLayoutError),
    /// Rotary policy or destination failure.
    #[error(transparent)]
    Rotary(#[from] RotaryTableError),
    /// The supplied source differs from the measured geometry.
    #[error("patch encoder table source differs from its measured geometry")]
    SourceGeometry,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Cells {
    start: usize,
    len: usize,
}
impl Cells {
    fn append(total: &mut usize, len: usize) -> Result<Self, SequenceLayoutError> {
        let start = *total;
        *total = total
            .checked_add(len)
            .ok_or(SequenceLayoutError::Overflow)?;
        Ok(Self { start, len })
    }
    fn get<T>(self, values: &[T]) -> &[T] {
        &values[self.start..self.start + self.len]
    }
}

/// Scalar ranges over two exact fixed buffers. Aliased identity permutations
/// and full-attention chunks occupy only their actual shared range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatchEncoderTableLayout {
    spec: PatchEncoderTableSpec,
    patches: i32,
    full: Cells,
    permutation: Cells,
    inverse: Cells,
    windows: Cells,
    learned: Cells,
    spatial: Cells,
    weights: Cells,
    frequencies: Cells,
    integers: usize,
    floats: usize,
}
impl PatchEncoderTableLayout {
    /// Measures borrowed immutable rows without constructing a grid or table.
    pub fn new(
        spec: PatchEncoderTableSpec,
        rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
        expected_patches: Option<i32>,
    ) -> Result<Self, PatchEncoderTableError> {
        let population = patch_grid_population(rows.clone(), spec.merge, expected_patches)?;
        let patches = population.patches as usize;
        let unit = spec
            .merge
            .checked_mul(spec.merge)
            .ok_or(SequenceLayoutError::Overflow)?;
        let groups = (population.patches / unit) as usize;
        let (mut integers, mut floats) = (0, 0);
        let full = Cells::append(&mut integers, population.frames)?;
        let permutation = Cells::append(&mut integers, groups)?;
        let (inverse, windows) = match spec.windows {
            PatchAttentionWindows::Full => (permutation, full),
            PatchAttentionWindows::Windowed {
                window_size,
                patch_size,
            } => {
                let window =
                    window_partition_population(rows.clone(), spec.merge, window_size, patch_size)?;
                (
                    Cells::append(&mut integers, window.groups)?,
                    Cells::append(&mut integers, window.windows)?,
                )
            }
        };
        let learned_count = match spec.positions {
            PatchPositionTableSpec::Gathered { height, width } => {
                gathered_position_population(rows.clone(), height, width)?;
                patches
            }
            PatchPositionTableSpec::Interpolated {
                height,
                width,
                mode,
                traversal,
            } => {
                interpolation_population(rows.clone(), height, width, mode, traversal)?;
                patches
                    .checked_mul(4)
                    .ok_or(SequenceLayoutError::Overflow)?
            }
        };
        let learned = Cells::append(&mut integers, learned_count)?;
        let spatial = Cells::append(
            &mut integers,
            patches
                .checked_mul(2)
                .ok_or(SequenceLayoutError::Overflow)?,
        )?;
        let weights = Cells::append(
            &mut floats,
            if matches!(spec.positions, PatchPositionTableSpec::Interpolated { .. }) {
                learned_count
            } else {
                0
            },
        )?;
        let frequencies = Cells::append(&mut floats, spec.rotary().frequency_count()?)?;
        // Reject actual Rust allocation-layout overflow before any destination.
        std::alloc::Layout::array::<i32>(integers).map_err(|_| SequenceLayoutError::Overflow)?;
        std::alloc::Layout::array::<f32>(floats).map_err(|_| SequenceLayoutError::Overflow)?;
        Ok(Self {
            spec,
            patches: population.patches,
            full,
            permutation,
            inverse,
            windows,
            learned,
            spatial,
            weights,
            frequencies,
            integers,
            floats,
        })
    }
    /// Exact i32 destination population.
    pub fn integer_count(self) -> usize {
        self.integers
    }
    /// Exact f32 destination population.
    pub fn float_count(self) -> usize {
        self.floats
    }
    /// Source patch sequence extent.
    pub fn patches(self) -> i32 {
        self.patches
    }
    /// Complete architecture-selected table equations.
    pub fn spec(self) -> PatchEncoderTableSpec {
        self.spec
    }

    /// Validates the complete source and both destinations before any write.
    /// The row iterator must borrow the same immutable source used to measure it.
    pub fn fill(
        self,
        rows: impl Iterator<Item = (i32, i32, i32)> + Clone,
        integers: &mut [i32],
        floats: &mut [f32],
    ) -> Result<(), PatchEncoderTableError> {
        if Self::new(self.spec, rows.clone(), Some(self.patches))? != self {
            return Err(PatchEncoderTableError::SourceGeometry);
        }
        for (required, actual) in [(self.integers, integers.len()), (self.floats, floats.len())] {
            if actual < required {
                return Err(SequenceLayoutError::DestinationTooShort { required, actual }.into());
            }
        }
        let (full, rest) = integers[..self.integers].split_at_mut(self.full.len);
        fill_attention_chunk_lengths(rows.clone(), full)?;
        let (permutation, rest) = rest.split_at_mut(self.permutation.len);
        let rest = match self.spec.windows {
            PatchAttentionWindows::Full => {
                for (index, cell) in permutation.iter_mut().enumerate() {
                    *cell = index as i32;
                }
                rest
            }
            PatchAttentionWindows::Windowed {
                window_size,
                patch_size,
            } => {
                let (inverse, rest) = rest.split_at_mut(self.inverse.len);
                let (windows, rest) = rest.split_at_mut(self.windows.len);
                fill_window_partition(
                    rows.clone(),
                    self.spec.merge,
                    window_size,
                    patch_size,
                    permutation,
                    inverse,
                    windows,
                )?;
                rest
            }
        };
        let (learned, spatial) = rest.split_at_mut(self.learned.len);
        let (weights, frequencies) = floats[..self.floats].split_at_mut(self.weights.len);
        match self.spec.positions {
            PatchPositionTableSpec::Gathered { height, width } => {
                fill_gathered_patch_positions(rows.clone(), height, width, learned)?;
            }
            PatchPositionTableSpec::Interpolated {
                height,
                width,
                mode,
                traversal,
            } => {
                fill_bilinear_planes(
                    rows.clone(),
                    height,
                    width,
                    mode,
                    traversal,
                    learned,
                    weights,
                )?;
            }
        }
        fill_spatial_patch_positions(rows, PatchTraversal::MergeMajor(self.spec.merge), spatial)?;
        self.spec.rotary().fill_frequencies(frequencies)?;
        Ok(())
    }
}

/// Borrowed immutable views of a complete table population. The owning source
/// must separately preserve identity, accounting and lifetime authority.
pub struct PatchEncoderTables<'a> {
    layout: PatchEncoderTableLayout,
    integers: &'a [i32],
    floats: &'a [f32],
}
impl<'a> PatchEncoderTables<'a> {
    /// Pairs exact completed storage with its scalar layout; this grants no authority.
    pub fn new(
        layout: PatchEncoderTableLayout,
        integers: &'a [i32],
        floats: &'a [f32],
    ) -> Result<Self, PatchEncoderTableError> {
        if integers.len() != layout.integers || floats.len() != layout.floats {
            return Err(PatchEncoderTableError::SourceGeometry);
        }
        Ok(Self {
            layout,
            integers,
            floats,
        })
    }
    /// Scalar checked layout.
    pub fn layout(&self) -> PatchEncoderTableLayout {
        self.layout
    }
    /// Full-attention segments.
    pub fn full_chunks(&self) -> &'a [i32] {
        self.layout.full.get(self.integers)
    }
    /// Window-attention segments (the same slice for full attention).
    pub fn window_chunks(&self) -> &'a [i32] {
        self.layout.windows.get(self.integers)
    }
    /// Merged execution order.
    pub fn permutation(&self) -> &'a [i32] {
        self.layout.permutation.get(self.integers)
    }
    /// Inverse merged order (the same slice for identity order).
    pub fn inverse(&self) -> &'a [i32] {
        self.layout.inverse.get(self.integers)
    }
    /// Direct indices, or four corner-major planes.
    pub fn learned_indices(&self) -> &'a [i32] {
        self.layout.learned.get(self.integers)
    }
    /// Four corner-major interpolation weight planes, empty for direct lookup.
    pub fn learned_weights(&self) -> &'a [f32] {
        self.layout.weights.get(self.floats)
    }
    /// Original merge-major y/x coordinate cells.
    pub fn spatial_positions(&self) -> &'a [i32] {
        self.layout.spatial.get(self.integers)
    }
    /// Borrowed rotary policy and its actual precomputed frequencies.
    pub fn rotary(&self) -> PreparedMultiAxisRotary<'_> {
        PreparedMultiAxisRotary::new(
            self.layout.spec.rotary(),
            self.layout.frequencies.get(self.floats),
        )
        .expect("checked complete rotary population")
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
