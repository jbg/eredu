//! Borrowed rotary source tables. Numerical input is not allocation authority.
use super::{MultiAxisRotaryLayout, RotaryAxisSpec};
use crate::{Error, Tensor};

/// Fixed geometry/capacity rejection from rotary table preparation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum RotaryTableError {
    /// At least one coordinate axis is required.
    #[error("multi-axis rotary requires at least one axis")]
    EmptyAxes,
    /// The wavelength base must be finite and positive.
    #[error("rotary base must be finite and positive")]
    InvalidBase,
    /// Axis dimensions violate the selected section policy.
    #[error("invalid rotary width {dimensions} for axis {axis}")]
    InvalidAxis {
        /// Coordinate-axis index.
        axis: usize,
        /// Invalid feature width.
        dimensions: i32,
    },
    /// Complete rotated width cannot be zero.
    #[error("multi-axis rotary total width must be positive")]
    ZeroDimensions,
    /// A count cannot be represented by the native signed geometry.
    #[error("rotary table dimensions overflowed")]
    Overflow,
    /// A supplied table does not have the required complete population.
    #[error("rotary table requires {required} cells, got {actual}")]
    Capacity {
        /// Complete required cell count.
        required: usize,
        /// Supplied cell count.
        actual: usize,
    },
    /// A backend has not implemented the borrowed table companion.
    #[error("prepared multi-axis rotary is not implemented by this backend")]
    UnsupportedBackend,
}

/// Complete borrowed counterpart of the ordinary owned rotary policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiAxisRotarySpecRef<'a> {
    /// Policies in position-ID axis order.
    pub axes: &'a [RotaryAxisSpec],
    /// Finite positive base wavelength.
    pub base: f32,
    /// Minimum position after offset and signed saturation.
    pub minimum_position: i32,
    /// Feature-axis arrangement.
    pub layout: MultiAxisRotaryLayout,
}
impl MultiAxisRotarySpecRef<'_> {
    /// Checks the entire policy without allocating or invoking a backend.
    pub fn dimensions(self) -> Result<i32, RotaryTableError> {
        if self.axes.is_empty() {
            return Err(RotaryTableError::EmptyAxes);
        }
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(RotaryTableError::InvalidBase);
        }
        i32::try_from(self.axes.len()).map_err(|_| RotaryTableError::Overflow)?;
        let mut total = 0i32;
        for (axis, spec) in self.axes.iter().enumerate() {
            if spec.dimensions < 0
                || spec.dimensions % 2 != 0
                || (spec.dimensions == 0
                    && self.layout != MultiAxisRotaryLayout::RoundRobinSections)
            {
                return Err(RotaryTableError::InvalidAxis {
                    axis,
                    dimensions: spec.dimensions,
                });
            }
            total = total
                .checked_add(spec.dimensions)
                .ok_or(RotaryTableError::Overflow)?;
        }
        if total == 0 {
            return Err(RotaryTableError::ZeroDimensions);
        }
        Ok(total)
    }

    /// Complete immutable frequency population for every selected layout.
    pub fn frequency_count(self) -> Result<usize, RotaryTableError> {
        usize::try_from(self.dimensions()? / 2).map_err(|_| RotaryTableError::Overflow)
    }

    /// Fills in axis order, or global half-width order for round-robin sections.
    /// Any policy/capacity error leaves the whole destination unchanged.
    pub fn fill_frequencies(self, output: &mut [f32]) -> Result<usize, RotaryTableError> {
        let dimensions = self.dimensions()?;
        let count = usize::try_from(dimensions / 2).map_err(|_| RotaryTableError::Overflow)?;
        if output.len() < count {
            return Err(RotaryTableError::Capacity {
                required: count,
                actual: output.len(),
            });
        }
        match self.layout {
            MultiAxisRotaryLayout::RoundRobinSections => {
                for (index, value) in output[..count].iter_mut().enumerate() {
                    // Preserve the original native f32 evaluation order.
                    *value = 1.0 / self.base.powf(2.0 * index as f32 / dimensions as f32);
                }
            }
            _ => {
                let mut at = 0;
                for axis in self.axes {
                    let n = axis.dimensions as usize / 2;
                    fill_rotary_axis_frequencies(
                        self.base,
                        axis.dimensions,
                        &mut output[at..at + n],
                    )?;
                    at += n;
                }
            }
        }
        Ok(count)
    }
}

/// Shared exact per-axis frequency equation used by ordinary and fixed storage.
pub fn fill_rotary_axis_frequencies(
    base: f32,
    dimensions: i32,
    output: &mut [f32],
) -> Result<usize, RotaryTableError> {
    if !base.is_finite() || base <= 0.0 {
        return Err(RotaryTableError::InvalidBase);
    }
    if dimensions <= 0 || dimensions % 2 != 0 {
        return Err(RotaryTableError::InvalidAxis {
            axis: 0,
            dimensions,
        });
    }
    let count = dimensions as usize / 2;
    if output.len() < count {
        return Err(RotaryTableError::Capacity {
            required: count,
            actual: output.len(),
        });
    }
    for (index, value) in output[..count].iter_mut().enumerate() {
        *value = 1.0 / base.powf((index * 2) as f32 / dimensions as f32);
    }
    Ok(count)
}

/// Checked borrowed numerical frequencies. This value grants no source identity,
/// account, native allocation, or execution permission.
#[derive(Debug, Clone, Copy)]
pub struct PreparedMultiAxisRotary<'a> {
    spec: MultiAxisRotarySpecRef<'a>,
    frequencies: &'a [f32],
}
impl<'a> PreparedMultiAxisRotary<'a> {
    /// Pairs an exact complete population with its policy; values remain input.
    pub fn new(
        spec: MultiAxisRotarySpecRef<'a>,
        frequencies: &'a [f32],
    ) -> Result<Self, RotaryTableError> {
        let required = spec.frequency_count()?;
        if frequencies.len() != required {
            return Err(RotaryTableError::Capacity {
                required,
                actual: frequencies.len(),
            });
        }
        Ok(Self { spec, frequencies })
    }
    /// Complete borrowed policy.
    pub fn spec(self) -> MultiAxisRotarySpecRef<'a> {
        self.spec
    }
    /// Complete immutable frequencies in the declared order.
    pub fn frequencies(self) -> &'a [f32] {
        self.frequencies
    }
}

/// Executes the explicit prepared companion without an owned-policy fallback.
pub fn multi_axis_rotary_embeddings_prepared<T: Tensor>(
    position_ids: &T,
    prepared: PreparedMultiAxisRotary<'_>,
    context: &T::Context,
) -> Result<(T, T), Error> {
    T::multi_axis_rotary_embeddings_prepared(position_ids, prepared, context)
}

/// Scalar execution of actual supplied frequencies for numerical backend tests.
/// The existing division-based reference remains independent and unchanged.
pub fn reference_multi_axis_rotary_embeddings_prepared(
    positions: &[i32],
    rows: usize,
    prepared: PreparedMultiAxisRotary<'_>,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let spec = prepared.spec;
    let dimensions = spec.dimensions().map_err(Error::backend_retained_source)? as usize;
    let axes = spec.axes.len();
    if rows == 0 || rows.checked_mul(axes) != Some(positions.len()) {
        return Err(Error::backend(
            "scalar multi-axis position geometry mismatch",
        ));
    }
    let count = rows
        .checked_mul(dimensions)
        .ok_or_else(|| Error::backend_retained_source(RotaryTableError::Overflow))?;
    let (mut cosine, mut sine) = (vec![0.; count], vec![0.; count]);
    let half = dimensions / 2;
    for row in 0..rows {
        let position = |axis: usize| {
            positions[row * axes + axis]
                .saturating_add(spec.axes[axis].position_offset)
                .max(spec.minimum_position) as f32
        };
        let mut write = |at: usize, angle: f32| {
            cosine[row * dimensions + at] = angle.cos();
            sine[row * dimensions + at] = angle.sin();
        };
        match spec.layout {
            MultiAxisRotaryLayout::RoundRobinSections => {
                for frequency in 0..half {
                    let candidate = frequency % axes;
                    let section = spec.axes[candidate].dimensions as usize / 2;
                    let axis =
                        if candidate != 0 && (frequency as u64) < section as u64 * axes as u64 {
                            candidate
                        } else {
                            0
                        };
                    let angle = position(axis) * prepared.frequencies[frequency];
                    write(frequency, angle);
                    write(frequency + half, angle);
                }
            }
            _ => {
                let mut frequency = 0;
                for (axis, policy) in spec.axes.iter().enumerate() {
                    let width = policy.dimensions as usize / 2;
                    for local in 0..width {
                        let angle = position(axis) * prepared.frequencies[frequency + local];
                        let (first, second) =
                            if spec.layout == MultiAxisRotaryLayout::IndependentAxes {
                                (2 * frequency + local, 2 * frequency + local + width)
                            } else {
                                (frequency + local, frequency + local + half)
                            };
                        write(first, angle);
                        write(second, angle);
                    }
                    frequency += width;
                }
            }
        }
    }
    Ok((cosine, sine))
}

#[cfg(test)]
#[path = "prepared_rotary_tests.rs"]
mod tests;
