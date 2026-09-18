//! Fixed effective-input reconstruction shared by metadata and native operators.
use crate::{GeneratedTensorSource, TensorElementType};

/// Exact callback provenance of a selected ordinary projection implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionInputObservationMechanism {
    /// The multiplication consumes its already-required borrowed input.
    Borrowed,
    /// GPU block-FP8 consumes compact quantized activations; an optional F32
    /// diagnostic reconstructs their effective values using the fixed program.
    BlockFp8Gpu,
}

/// Invalid fixed-program geometry or unavailable selected source provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ProjectionObservationError {
    /// Empty, nonpositive or incompatible source axes.
    #[error("invalid block-FP8 projection-input geometry")]
    Geometry,
    /// Native grid, padded shape or logical quota exceeds its representation.
    #[error("block-FP8 projection-input arithmetic overflow")]
    Overflow,
    /// The selected provider did not describe the transformed input mechanism.
    #[error("selected projection-input observation mechanism is unavailable")]
    Unavailable,
}

/// Checked prototype geometry for the fixed 128-value activation-block program.
/// This borrows actual shape metadata and grants no allocation or source custody.
#[derive(Debug, Clone, Copy)]
pub struct BlockFp8InputReconstructionPlan<'a> {
    shape: &'a [i32],
    rows: i32,
    width: i32,
    scale_columns: i32,
    padded_width: i32,
    elements: u64,
}
impl<'a> BlockFp8InputReconstructionPlan<'a> {
    /// Checks positive axes and every native i32 grid/padded-shape extent.
    pub fn new(shape: &'a [i32]) -> Result<Self, ProjectionObservationError> {
        let width = *shape.last().ok_or(ProjectionObservationError::Geometry)?;
        if shape.iter().any(|&n| n <= 0) {
            return Err(ProjectionObservationError::Geometry);
        }
        let elements = shape.iter().try_fold(1u64, |n, &axis| {
            n.checked_mul(axis as u64)
                .ok_or(ProjectionObservationError::Overflow)
        })?;
        let elements_i32 =
            i32::try_from(elements).map_err(|_| ProjectionObservationError::Overflow)?;
        let rows = elements_i32 / width;
        let scale_columns = i32::try_from((width as u64).div_ceil(128))
            .map_err(|_| ProjectionObservationError::Overflow)?;
        let padded_width = scale_columns
            .checked_mul(128)
            .ok_or(ProjectionObservationError::Overflow)?;
        rows.checked_mul(padded_width)
            .ok_or(ProjectionObservationError::Overflow)?;
        Ok(Self {
            shape,
            rows,
            width,
            scale_columns,
            padded_width,
            elements,
        })
    }
    /// Original prototype axes; reconstruction restores these exact axes.
    pub fn shape(self) -> &'a [i32] {
        self.shape
    }
    /// Flattened independent activation rows.
    pub fn rows(self) -> i32 {
        self.rows
    }
    /// Original feature width.
    pub fn width(self) -> i32 {
        self.width
    }
    /// One scale per complete or partial 128-value block.
    pub fn scale_columns(self) -> i32 {
        self.scale_columns
    }
    /// Repeated scale allocation includes complete blocks, including padding.
    pub fn padded_width(self) -> i32 {
        self.padded_width
    }
    /// Exact compact quantized-activation shape.
    pub fn values_shape(self) -> [i32; 2] {
        [self.rows, self.width]
    }
    /// Exact compact activation-scale shape.
    pub fn scales_shape(self) -> [i32; 2] {
        [self.rows, self.scale_columns]
    }
    /// Validates the actual already-required quantized operands.
    pub fn validate_operands(
        self,
        values: &[i32],
        scales: &[i32],
    ) -> Result<(), ProjectionObservationError> {
        if values != self.values_shape() || scales != self.scales_shape() {
            return Err(ProjectionObservationError::Geometry);
        }
        Ok(())
    }
    /// Logical buffers of the fixed reconstruction: padded repeated scales,
    /// decoded values, trimmed scales and their product. Shape views add no
    /// logical buffer. Physical backing, aliasing, controls and completion owners
    /// are priced separately by the selected native operation program.
    pub fn logical_capture_source(
        self,
    ) -> Result<GeneratedTensorSource, ProjectionObservationError> {
        let creation_bytes = (self.rows as u64)
            .checked_mul(self.padded_width as u64)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| {
                self.elements
                    .checked_mul(12)
                    .and_then(|values| n.checked_add(values))
            })
            .ok_or(ProjectionObservationError::Overflow)?;
        Ok(GeneratedTensorSource {
            creation_bytes,
            element_type: Some(TensorElementType::F32),
        })
    }
}

/// Realizes only the fixed reconstruction operations. Implementations borrow
/// actual compact U8 values and F32 scales already required by the projection.
/// No callback here grants allocation, native completion, or funding authority.
pub trait BlockFp8InputReconstructionMechanism {
    /// Native or metadata value.
    type Value;
    /// Original realization error.
    type Error;
    /// Inserts the repetition axis after the scale-column axis.
    fn expand_scales(&self) -> Result<Self::Value, Self::Error>;
    /// Broadcasts each scale to a complete 128-value block.
    fn broadcast_scales(&self, expanded: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Materializes or aliases the repeated `[rows,padded_width]` scale matrix.
    fn flatten_scales(&self, broadcasted: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Converts actual compact E4M3 bytes to F32, without applying scales.
    fn decode_values(&self) -> Result<Self::Value, Self::Error>;
    /// Borrows the leading original-width scale columns.
    fn trim_scales(&self, repeated: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Multiplies decoded values by their actual repeated scales.
    fn multiply(
        &self,
        values: &Self::Value,
        scales: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
    /// Restores exactly the borrowed prototype shape.
    fn restore_shape(&self, product: &Self::Value) -> Result<Self::Value, Self::Error>;
}

/// Executes the same ordered reconstruction in native and workspace realizations.
/// Every intermediate remains owned through the final operation; the caller's
/// original native work scope governs completion and failed-graph retirement.
pub fn reconstruct_block_fp8_input<M: BlockFp8InputReconstructionMechanism>(
    mechanism: M,
) -> Result<M::Value, M::Error> {
    reconstruct_block_fp8_input_retained(mechanism, &mut |_| Ok(()))
}

/// Execute the same fixed program, retaining each actual output before the next
/// fallible operation. Source inputs stay borrowed from the mechanism; callers
/// visit them separately before generation. No completion or funding is implied.
pub fn reconstruct_block_fp8_input_retained<M: BlockFp8InputReconstructionMechanism>(
    mechanism: M,
    retain: &mut dyn FnMut(&M::Value) -> Result<(), M::Error>,
) -> Result<M::Value, M::Error> {
    let expanded = mechanism.expand_scales()?;
    retain(&expanded)?;
    let broadcasted = mechanism.broadcast_scales(&expanded)?;
    retain(&broadcasted)?;
    let repeated = mechanism.flatten_scales(&broadcasted)?;
    retain(&repeated)?;
    let decoded = mechanism.decode_values()?;
    retain(&decoded)?;
    let trimmed = mechanism.trim_scales(&repeated)?;
    retain(&trimmed)?;
    let product = mechanism.multiply(&decoded, &trimmed)?;
    retain(&product)?;
    let output = mechanism.restore_shape(&product)?;
    retain(&output)?;
    Ok(output)
}

#[cfg(test)]
mod tests;
