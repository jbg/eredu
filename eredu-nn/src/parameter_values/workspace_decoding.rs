//! Descriptive realization of the selected effective parameter decoder.
use super::{ParameterDecoding, ParameterDecodingMechanism};
use crate::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceMetadataError,
        WorkspaceOperationKind, WorkspaceTensor,
    },
    Error, LinearFormat, Tensor,
};

/// The actual primary and companion source views, projected into one context.
/// Logical geometry comes from the retained architecture declaration. Native
/// mechanisms independently qualify storage, constructor and completion facts.
pub struct WorkspaceParameterDecoding<'a> {
    weight: &'a WorkspaceTensor,
    scale: Option<&'a WorkspaceTensor>,
    bias: Option<&'a WorkspaceTensor>,
    shape: &'a [i32],
    context: &'a WorkspaceContext,
}
impl<'a> WorkspaceParameterDecoding<'a> {
    /// Borrow the exact selected source slots without allocating native tensors.
    pub fn new(
        weight: &'a WorkspaceTensor,
        scale: Option<&'a WorkspaceTensor>,
        bias: Option<&'a WorkspaceTensor>,
        shape: &'a [i32],
        context: &'a WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>)>())?;
        context.validate_values(Some(weight).into_iter().chain(scale).chain(bias))?;
        if shape.is_empty() || shape.iter().any(|extent| *extent <= 0) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        Ok(Self {
            weight,
            scale,
            bias,
            shape,
            context,
        })
    }
    fn decode(
        &self,
        inputs: &[&WorkspaceTensor],
        decoding: ParameterDecoding,
    ) -> Result<WorkspaceTensor, Error> {
        let mut outputs = self.context.metadata_vec(1)?;
        outputs.push(self.context.layout(self.shape, WorkspaceDtype::Float32)?);
        let mut values = self.context.execute(
            WorkspaceOperationKind::ParameterDecode(decoding),
            inputs,
            outputs,
        )?;
        values
            .pop()
            .ok_or_else(|| WorkspaceMetadataError::Unqualified.into())
    }
}
impl ParameterDecodingMechanism for WorkspaceParameterDecoding<'_> {
    type Value = WorkspaceTensor;
    type Error = Error;
    fn weight(&self) -> Result<&WorkspaceTensor, Error> {
        Ok(self.weight)
    }
    fn is_floating(&self, value: &WorkspaceTensor) -> bool {
        value.layout().dtype() == WorkspaceDtype::Float32
    }
    fn alias(&self, value: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        Ok(value.clone())
    }
    fn scale(&self) -> Result<&WorkspaceTensor, Error> {
        self.scale
            .ok_or_else(|| WorkspaceMetadataError::Unqualified.into())
    }
    fn affine_bias(&self) -> Result<&WorkspaceTensor, Error> {
        self.bias
            .ok_or_else(|| WorkspaceMetadataError::Unqualified.into())
    }
    fn cast_f32(&self, value: &WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.cast_floating(WorkspaceFloatingType::Float32, self.context)
    }
    fn affine(
        &self,
        weight: &WorkspaceTensor,
        scale: &WorkspaceTensor,
        bias: &WorkspaceTensor,
        config: eredu_checkpoint::AffineQuantization,
    ) -> Result<WorkspaceTensor, Error> {
        self.decode(
            &[weight, scale, bias],
            ParameterDecoding {
                format: LinearFormat::Affine(config),
                row_layout: crate::LinearRowLayout::Contiguous,
            },
        )
    }
    fn mx_fp4(
        &self,
        weight: &WorkspaceTensor,
        scale: &WorkspaceTensor,
    ) -> Result<WorkspaceTensor, Error> {
        self.decode(
            &[weight, scale],
            ParameterDecoding {
                format: LinearFormat::MxFp4,
                row_layout: crate::LinearRowLayout::Contiguous,
            },
        )
    }
    fn block_fp8(
        &self,
        weight: &WorkspaceTensor,
        scale: &WorkspaceTensor,
        decoding: ParameterDecoding,
    ) -> Result<WorkspaceTensor, Error> {
        self.decode(&[weight, scale], decoding)
    }
    fn gguf(
        &self,
        weight: &WorkspaceTensor,
        decoding: ParameterDecoding,
    ) -> Result<WorkspaceTensor, Error> {
        self.decode(&[weight], decoding)
    }
    fn restore_shape(&self, value: WorkspaceTensor) -> Result<WorkspaceTensor, Error> {
        value.reshape(self.shape, self.context)
    }
    fn invalid_dense(&self) -> Error {
        WorkspaceMetadataError::Unqualified.into()
    }
}
