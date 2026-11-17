//! Checked logical tensor extents independent of storage and execution.

/// Counts row-major elements while rejecting an overflowing nonzero extent.
pub fn checked_elements(shape: &[u64]) -> Option<u64> {
    let nonzero = shape
        .iter()
        .copied()
        .filter(|&dimension| dimension != 0)
        .try_fold(1u64, u64::checked_mul)?;
    Some(if shape.contains(&0) { 0 } else { nonzero })
}
