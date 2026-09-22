//! Prepared native handle publication over the neutral participant transaction.
use crate::composition::mlx::replicated_text::ErasedReplicatedTextExecutable;
use crate::{
    backend::{
        error::Error, managed_memory::PreparedPublicationClone,
        runtime::execution::generic::MlxParameterPreparation,
    },
    MlxTensor,
};
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::{
    parameter_operations::{
        ParameterPublicationError, ParameterReplacementValues,
        PreparedParameterPublication,
    },
    working_memory::InferenceExecutionIdentity,
};
use std::mem::size_of;

pub(super) struct PreparedNativeParameterPublication {
    prepared: PreparedParameterPublication<MlxTensor>,
    execution: InferenceExecutionIdentity,
    context: WorkspaceContext,
}
impl PreparedNativeParameterPublication {
    pub(super) fn prepare(
        model: &mut dyn ErasedReplicatedTextExecutable,
        values: ParameterReplacementValues<MlxTensor>,
        active: bool,
        preparation: &MlxParameterPreparation<'_>,
    ) -> Result<Self, Error> {
        let context = preparation
            .mechanism
            .context(preparation.funding.clone())
            .map_err(|cause| Error::Neural(cause.into()))?;
        context
            .charge_metadata(size_of::<Self>())
            .map_err(|cause| Error::Neural(cause.into()))?;
        validate_execution(model, preparation.execution)?;
        let mut prepared = None;
        model.with_parameter_publication(&context, &mut |visit| {
            prepared = Some(
                PreparedParameterPublication::prepare(
                    values.clone(),
                    active,
                    visit,
                    |value| {
                        let clone =
                            PreparedPublicationClone::prepare(preparation.environment.pool())
                                .and_then(|slot| slot.fill(value.as_array()))
                                .map_err(|cause| context.metadata_source(cause))?;
                        Ok(MlxTensor::from_array(clone))
                    },
                    &context,
                    preparation.funding.clone(),
                )
                .map_err(|cause| failure(&context, cause))?,
            );
            Ok(())
        })?;
        Ok(Self {
            prepared: prepared.ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))?,
            execution: preparation.execution.clone(),
            context,
        })
    }
    pub(super) fn exchange_reset(
        &mut self,
        model: &mut dyn ErasedReplicatedTextExecutable,
        reset: &mut dyn std::any::Any,
    ) -> Result<(), Error> {
        validate_execution(model, &self.execution)?;
        model.commit_parameter_publication(&mut self.prepared, reset, &self.context)
    }
}
fn validate_execution(
    model: &dyn ErasedReplicatedTextExecutable,
    expected: &InferenceExecutionIdentity,
) -> Result<(), Error> {
    if !model
        .inference_execution_identity()
        .same_execution(expected)
    {
        return Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ));
    }
    Ok(())
}
fn failure(context: &WorkspaceContext, cause: ParameterPublicationError<Error>) -> Error {
    MlxParameterPreparation::publication_failure(context, cause)
}
