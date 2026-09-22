//! Fixed effective-parameter read and contraction sequencing.
//!
//! Numerical and metadata realizations use the same worker. These contracts
//! establish neither source identity nor storage, admission or completion authority.

mod workspace;
pub use workspace::WorkspaceParameterValues;
mod workspace_decoding;
pub use workspace_decoding::WorkspaceParameterDecoding;
mod decoding;
pub use decoding::{decode_parameter, ParameterDecoding, ParameterDecodingMechanism};

/// Operations shared by a selected effective parameter read and projection.
pub trait ParameterValueMechanism {
    /// Numerical or metadata tensor value.
    type Value;
    /// Completed host output or the metadata realization's completion result.
    type Output;
    /// Original realization failure.
    type Error;
    /// Select the admitted rectangular region of the borrowed effective source.
    fn select(&self) -> Result<Self::Value, Self::Error>;
    /// Convert the selected values to F32 using the selected realization.
    fn cast_f32(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Materialize row-major storage when the actual layout requires it.
    fn contiguous(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Complete this actual output and read into independently funded host storage.
    /// Metadata realizations record the same completion boundary without reading.
    fn complete_and_read(&self, value: Self::Value) -> Result<Self::Output, Self::Error>;
}

/// Runs selection, floating conversion, row-major materialization and completion.
pub fn read_parameter<M: ParameterValueMechanism>(mechanism: M) -> Result<M::Output, M::Error> {
    let selected = mechanism.select()?;
    let selected = mechanism.cast_f32(selected)?;
    let output = mechanism.contiguous(selected)?;
    mechanism.complete_and_read(output)
}

/// Additional operations of the existing effective-parameter contraction.
pub trait ParameterProjectionMechanism: ParameterValueMechanism {
    /// Move the selected contraction axis to the final position.
    fn transpose_weights(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Present the selected values as rows of the checked contraction width.
    fn reshape_weights(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Construct the typed F32 coefficient matrix from the actual borrowed host
    /// source. A metadata initialization equation is not construction permission.
    fn coefficients(&self) -> Result<Self::Value, Self::Error>;
    /// Transpose directions into the matrix multiplication's right-hand operand.
    fn transpose_coefficients(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Contract the two actual prepared operands.
    fn matmul(
        &self,
        weights: &Self::Value,
        coefficients: &Self::Value,
    ) -> Result<Self::Value, Self::Error>;
}

/// Runs the fixed contraction and its output completion through one realization.
pub fn project_parameter<M: ParameterProjectionMechanism>(
    mechanism: M,
) -> Result<M::Output, M::Error> {
    let selected = mechanism.select()?;
    let selected = mechanism.cast_f32(selected)?;
    let weights = mechanism.transpose_weights(selected)?;
    let weights = mechanism.contiguous(weights)?;
    let weights = mechanism.reshape_weights(weights)?;
    let coefficients = mechanism.coefficients()?;
    let coefficients = mechanism.transpose_coefficients(coefficients)?;
    let coefficients = mechanism.contiguous(coefficients)?;
    let output = mechanism.matmul(&weights, &coefficients)?;
    let output = mechanism.contiguous(output)?;
    mechanism.complete_and_read(output)
}

mod update;
pub use update::{update_parameter, ParameterUpdateMechanism, WorkspaceParameterUpdate};
