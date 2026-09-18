//! Shared allocation-free shape equations with exact caller-owned destinations.
use super::Error;

/// Fixed failure from a shape equation or its destination contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceShapeError {
    /// Trailing dimensions cannot broadcast under the ordinary tensor rules.
    #[error("incompatible workspace broadcast shapes")]
    IncompatibleBroadcast,
    /// Matrix multiplication has a scalar operand.
    #[error("workspace matmul requires ranked inputs")]
    MatmulRank,
    /// The two contraction dimensions differ.
    #[error("workspace matmul reduction extents differ")]
    MatmulReduction,
    /// A destination must hold exactly the result's dimensions.
    #[error("workspace shape destination rank mismatch: expected {expected}, got {actual}")]
    DestinationRank { expected: usize, actual: usize },
}

impl WorkspaceShapeError {
    pub(super) fn into_ordinary(self) -> Error {
        match self {
            Self::IncompatibleBroadcast => {
                Error::backend("incompatible workspace broadcast shapes")
            }
            Self::MatmulRank => Error::backend("workspace matmul requires ranked inputs"),
            Self::MatmulReduction => Error::backend("workspace matmul reduction extents differ"),
            Self::DestinationRank { .. } => {
                Error::backend("workspace shape destination rank mismatch")
            }
        }
    }
}

/// Fixed geometry failure for row scatter, independent of mask values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceScatterShapeError {
    /// A row-selection mask must exactly name the leading input dimensions.
    #[error("workspace scatter mask must match the input's leading dimensions")]
    MaskPrefix,
    /// Source rows cannot broadcast to the unmasked input dimensions.
    #[error("workspace scatter source cannot broadcast to the input row dimensions")]
    SourceRows,
}

/// Validates row-scatter geometry without inspecting mask values or allocating.
/// The mask exactly matches the input prefix. Source dimensions broadcast to
/// the remaining row dimensions, optionally preceded by a source-row count.
/// Whether that count covers the true mask entries is a numerical precondition
/// and cannot be established by a metadata-only trace.
pub fn validate_masked_scatter_shapes(
    input: &[i32], mask: &[i32], source: &[i32],
) -> Result<(), WorkspaceScatterShapeError> {
    if mask.len() > input.len() || input[..mask.len()] != *mask {
        return Err(WorkspaceScatterShapeError::MaskPrefix);
    }
    let row = &input[mask.len()..];
    if source.len() > row.len() + 1 {
        return Err(WorkspaceScatterShapeError::SourceRows);
    }
    let source_row = if source.len() > row.len() { &source[1..] } else { source };
    let shape = WorkspaceBroadcastShape::new(row, source_row)
        .map_err(|_| WorkspaceScatterShapeError::SourceRows)?;
    if !shape.dimensions().eq(row.iter().copied()) {
        return Err(WorkspaceScatterShapeError::SourceRows);
    }
    Ok(())
}

/// A broadcast equation borrowing both operands, without a result allocation.
///
/// This validates compatibility, exactly as the ordinary tensor shape helper
/// does. Extent signs and byte arithmetic remain the separate layout contract;
/// callers use `WorkspaceLayoutView` to validate the filled layout. This plan
/// neither allocates storage nor grants execution or memory authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceBroadcastShape<'a> {
    left: &'a [i32],
    right: &'a [i32],
}

fn broadcast_extent(left: i32, right: i32) -> Result<i32, WorkspaceShapeError> {
    if left == right || right == 1 {
        Ok(left)
    } else if left == 1 {
        Ok(right)
    } else {
        Err(WorkspaceShapeError::IncompatibleBroadcast)
    }
}

impl<'a> WorkspaceBroadcastShape<'a> {
    /// Checks trailing-axis compatibility, including singleton and zero axes.
    pub fn new(left: &'a [i32], right: &'a [i32]) -> Result<Self, WorkspaceShapeError> {
        let plan = Self { left, right };
        for trailing in 0..plan.rank() {
            plan.trailing_dimension(trailing)?;
        }
        Ok(plan)
    }

    /// Exact number of dimensions required by the result destination.
    pub fn rank(self) -> usize {
        self.left.len().max(self.right.len())
    }

    fn trailing_dimension(self, trailing: usize) -> Result<i32, WorkspaceShapeError> {
        let left = self
            .left
            .len()
            .checked_sub(trailing + 1)
            .map_or(1, |i| self.left[i]);
        let right = self
            .right
            .len()
            .checked_sub(trailing + 1)
            .map_or(1, |i| self.right[i]);
        broadcast_extent(left, right)
    }

    /// Result dimensions in logical order, borrowing the original operands.
    pub fn dimensions(self) -> impl ExactSizeIterator<Item = i32> + 'a {
        (0..self.rank()).rev().map(move |trailing| {
            self.trailing_dimension(trailing)
                .expect("the immutable broadcast equation was checked")
        })
    }

    /// Fills an exact destination, leaving it unchanged on a rank mismatch.
    pub fn write_into(self, destination: &mut [i32]) -> Result<(), WorkspaceShapeError> {
        write_dimensions(self.rank(), self.dimensions(), destination)
    }
}

/// Matrix multiplication shape, including vector promotion and batch broadcast.
/// Both source shapes remain borrowed; no promoted operand arrays are created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceMatmulShape<'a> {
    batch: WorkspaceBroadcastShape<'a>,
    rows: Option<i32>,
    columns: Option<i32>,
}

impl<'a> WorkspaceMatmulShape<'a> {
    /// Checks rank, then contraction width, then trailing batch compatibility,
    /// preserving the ordinary matrix multiplication diagnostic order.
    pub fn new(left: &'a [i32], right: &'a [i32]) -> Result<Self, WorkspaceShapeError> {
        if left.is_empty() || right.is_empty() {
            return Err(WorkspaceShapeError::MatmulRank);
        }
        let left_vector = left.len() == 1;
        let right_vector = right.len() == 1;
        let right_reduction = if right_vector {
            right[0]
        } else {
            right[right.len() - 2]
        };
        if left[left.len() - 1] != right_reduction {
            return Err(WorkspaceShapeError::MatmulReduction);
        }
        let batch = WorkspaceBroadcastShape::new(
            &left[..left.len().saturating_sub(2)],
            &right[..right.len().saturating_sub(2)],
        )?;
        Ok(Self {
            batch,
            rows: (!left_vector).then(|| left[left.len() - 2]),
            columns: (!right_vector).then(|| right[right.len() - 1]),
        })
    }

    /// Exact result rank; a vector dot product has rank zero.
    pub fn rank(self) -> usize {
        self.batch.rank() + usize::from(self.rows.is_some()) + usize::from(self.columns.is_some())
    }

    /// Result dimensions, without promoted input or output storage.
    pub fn dimensions(self) -> impl ExactSizeIterator<Item = i32> + 'a {
        let batch_rank = self.batch.rank();
        (0..self.rank()).map(move |axis| {
            if axis < batch_rank {
                self.batch
                    .trailing_dimension(batch_rank - axis - 1)
                    .expect("the immutable batch equation was checked")
            } else if axis == batch_rank && self.rows.is_some() {
                self.rows.expect("row dimension is present")
            } else {
                self.columns.expect("column dimension is present")
            }
        })
    }

    /// Fills an exact destination, leaving it unchanged on a rank mismatch.
    pub fn write_into(self, destination: &mut [i32]) -> Result<(), WorkspaceShapeError> {
        write_dimensions(self.rank(), self.dimensions(), destination)
    }
}

fn write_dimensions(
    expected: usize,
    dimensions: impl Iterator<Item = i32>,
    destination: &mut [i32],
) -> Result<(), WorkspaceShapeError> {
    if destination.len() != expected {
        return Err(WorkspaceShapeError::DestinationRank {
            expected,
            actual: destination.len(),
        });
    }
    for (target, value) in destination.iter_mut().zip(dimensions) {
        *target = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
