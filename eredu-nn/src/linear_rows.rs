//! Physical row-block boundaries of an encoded projection.

use crate::Error;

/// Independent block origins within a projection's output-row axis.
/// Equal partitions retain their meaning when each partition is sliced to the
/// same rank-local width, as with component-major fused gate/up projections.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum LinearRowLayout {
    /// One block origin for the complete output axis.
    #[default]
    Contiguous,
    /// Restart block numbering in each equal contiguous row partition.
    EqualPartitions(std::num::NonZeroUsize),
}

impl LinearRowLayout {
    /// Declares a positive count of independently blocked row partitions.
    pub fn equal_partitions(count: usize) -> Result<Self, Error> {
        std::num::NonZeroUsize::new(count)
            .map(|count| {
                if count.get() == 1 {
                    Self::Contiguous
                } else {
                    Self::EqualPartitions(count)
                }
            })
            .ok_or_else(|| Error::backend("linear row partition count must be positive"))
    }

    /// Number of independent row partitions.
    pub const fn partitions(self) -> usize {
        match self {
            Self::Contiguous => 1,
            Self::EqualPartitions(count) => count.get(),
        }
    }

    /// Exact rows in each partition; rejects empty or unequal geometry.
    pub fn rows_per_partition(self, rows: usize) -> Result<usize, Error> {
        let parts = self.partitions();
        if rows == 0 || !rows.is_multiple_of(parts) {
            return Err(Error::backend(
                "linear rows do not form positive equal partitions",
            ));
        }
        Ok(rows / parts)
    }

    /// Total stored scale rows, retaining the final partial block of each partition.
    pub fn scale_rows(self, rows: usize, block: usize) -> Result<usize, Error> {
        if block == 0 {
            return Err(Error::backend("linear row block width is zero"));
        }
        self.rows_per_partition(rows)?
            .div_ceil(block)
            .checked_mul(self.partitions())
            .ok_or_else(|| Error::backend("linear scale row count overflows"))
    }

    /// Maps a complete block or partition boundary to its scale coordinate.
    pub fn block_boundary(
        self,
        rows: usize,
        block: usize,
        boundary: usize,
    ) -> Result<usize, Error> {
        let width = self.rows_per_partition(rows)?;
        if block == 0 || boundary > rows || !(boundary % width).is_multiple_of(block) {
            return Err(Error::backend(
                "linear row boundary splits an encoded block",
            ));
        }
        (boundary / width)
            .checked_mul(width.div_ceil(block))
            .and_then(|offset| offset.checked_add((boundary % width) / block))
            .ok_or_else(|| Error::backend("linear scale boundary overflows"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_row_tails_preserve_scale_origins_after_partitioning() {
        let layout = LinearRowLayout::equal_partitions(2).unwrap();
        assert_eq!(layout.scale_rows(518, 128).unwrap(), 6);
        for (row, scale) in [
            (0, 0),
            (128, 1),
            (256, 2),
            (259, 3),
            (387, 4),
            (515, 5),
            (518, 6),
        ] {
            assert_eq!(layout.block_boundary(518, 128, row).unwrap(), scale);
        }
        assert_eq!(layout.scale_rows(6, 128).unwrap(), 2);
        assert_eq!(layout.scale_rows(512, 128).unwrap(), 4);
        assert!(layout.block_boundary(518, 128, 258).is_err());
        assert!(layout.block_boundary(518, 128, 519).is_err());
        assert!(layout.scale_rows(517, 128).is_err());
        assert!(layout.scale_rows(0, 128).is_err());
        assert!(layout.scale_rows(518, 0).is_err());
        assert!(LinearRowLayout::equal_partitions(0).is_err());
        assert_eq!(
            LinearRowLayout::Contiguous
                .scale_rows(usize::MAX, 128)
                .unwrap(),
            usize::MAX.div_ceil(128)
        );
    }
}
