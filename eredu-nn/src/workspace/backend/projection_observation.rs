use super::*;
use crate::{
    reconstruct_block_fp8_input_retained, BlockFp8InputReconstructionMechanism,
    BlockFp8InputReconstructionPlan, ProjectionInputObservationMechanism,
    ProjectionObservationError,
};

impl WorkspaceLinear {
    pub(super) fn forward_observed(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
        observer: Option<&mut dyn crate::ProjectionInputObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        let Some(observer) = observer else {
            return self.forward(input, context);
        };
        match context.projection_input_observation_mechanism(&self.spec.format)? {
            Some(ProjectionInputObservationMechanism::Borrowed) => {
                observer.observe(input)?;
                self.forward(input, context)
            }
            Some(ProjectionInputObservationMechanism::BlockFp8Gpu) => {
                if !matches!(self.spec.format.encoding(), LinearFormat::E4M3BlockFp8(_))
                    || input.shape().last() != Some(&self.spec.input)
                    || input.layout().dtype() != WorkspaceDtype::Float32
                {
                    return Err(context.metadata_source(ProjectionObservationError::Geometry));
                }
                let plan = BlockFp8InputReconstructionPlan::new(input.shape())
                    .map_err(|cause| context.metadata_source(cause))?;
                let mut inputs = context.metadata_vec(
                    self.parameters
                        .len()
                        .checked_add(3)
                        .ok_or(WorkspaceMetadataError::Overflow)?,
                )?;
                inputs.push(input);
                inputs.extend(self.parameters.iter().map(Parameter::as_ref));
                let mut layouts = context.metadata_vec(2)?;
                layouts.extend([
                    context.layout(&plan.values_shape(), WorkspaceDtype::Uint8)?,
                    context.layout(&plan.scales_shape(), WorkspaceDtype::Float32)?,
                ]);
                let mut prepared = context.execute(
                    WorkspaceOperationKind::ProjectionPrepare(
                        context.clone_metadata(&self.spec.format)?,
                    ),
                    &inputs,
                    layouts,
                )?;
                let scales = prepared.pop().expect("declared scale output");
                let values = prepared.pop().expect("declared packed output");
                let source = plan
                    .logical_capture_source()
                    .map_err(|cause| context.metadata_source(cause))?;
                observer.observe_generated_retained(
                    input,
                    &source,
                    &mut Reconstruction {
                        plan,
                        values: &values,
                        scales: &scales,
                        context,
                    },
                )?;
                inputs.extend([&values, &scales]);
                let mut shape = context.metadata_vec(input.shape().len())?;
                shape.extend_from_slice(input.shape());
                *shape.last_mut().expect("validated rank") = self.spec.output;
                WorkspaceTensor::operation(
                    WorkspaceOperationKind::ProjectionFinish(
                        context.clone_metadata(&self.spec.format)?,
                    ),
                    &inputs,
                    &shape,
                    WorkspaceDtype::Float32,
                    context,
                )
            }
            None => Err(context.metadata_source(ProjectionObservationError::Unavailable)),
        }
    }
}

struct Reconstruction<'a> {
    plan: BlockFp8InputReconstructionPlan<'a>,
    values: &'a WorkspaceTensor,
    scales: &'a WorkspaceTensor,
    context: &'a WorkspaceContext,
}
impl crate::RetainedGeneratedTensorFactory<WorkspaceTensor, Error> for Reconstruction<'_> {
    fn program(&self) -> crate::GeneratedTensorProgram<'_> {
        crate::GeneratedTensorProgram::BlockFp8Input(self.plan)
    }
    fn visit_sources(
        &mut self,
        retain: &mut dyn FnMut(
            crate::GeneratedTensorSourceRole,
            &WorkspaceTensor,
        ) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.plan
            .validate_operands(self.values.shape(), self.scales.shape())
            .map_err(|cause| self.context.metadata_source(cause))?;
        retain(crate::GeneratedTensorSourceRole::CompactValues, self.values)?;
        retain(crate::GeneratedTensorSourceRole::BlockScales, self.scales)
    }
    fn generate(
        &mut self,
        retain: &mut dyn FnMut(&WorkspaceTensor) -> Result<(), Error>,
    ) -> Result<WorkspaceTensor, Error> {
        self.plan
            .validate_operands(self.values.shape(), self.scales.shape())
            .map_err(|cause| self.context.metadata_source(cause))?;
        reconstruct_block_fp8_input_retained(
            Reconstruction {
                plan: self.plan,
                values: self.values,
                scales: self.scales,
                context: self.context,
            },
            retain,
        )
    }
}

impl BlockFp8InputReconstructionMechanism for Reconstruction<'_> {
    type Value = WorkspaceTensor;
    type Error = Error;
    fn expand_scales(&self) -> Result<Self::Value, Error> {
        self.scales.expand_dims(2, self.context)
    }
    fn broadcast_scales(&self, expanded: &Self::Value) -> Result<Self::Value, Error> {
        expanded.broadcast_to(
            &[self.plan.rows(), self.plan.scale_columns(), 128],
            self.context,
        )
    }
    fn flatten_scales(&self, broadcasted: &Self::Value) -> Result<Self::Value, Error> {
        broadcasted.reshape(&[self.plan.rows(), self.plan.padded_width()], self.context)
    }
    fn decode_values(&self) -> Result<Self::Value, Error> {
        WorkspaceTensor::operation(
            WorkspaceOperationKind::BlockFp8ActivationDecode,
            &[self.values],
            &self.plan.values_shape(),
            WorkspaceDtype::Float32,
            self.context,
        )
    }
    fn trim_scales(&self, repeated: &Self::Value) -> Result<Self::Value, Error> {
        repeated.index(
            &[
                crate::Index::Full,
                crate::Index::Range(0, self.plan.width()),
            ],
            self.context,
        )
    }
    fn multiply(&self, values: &Self::Value, scales: &Self::Value) -> Result<Self::Value, Error> {
        values.multiply(scales, self.context)
    }
    fn restore_shape(&self, product: &Self::Value) -> Result<Self::Value, Error> {
        product.reshape(self.plan.shape(), self.context)
    }
}

#[cfg(test)]
mod tests;
