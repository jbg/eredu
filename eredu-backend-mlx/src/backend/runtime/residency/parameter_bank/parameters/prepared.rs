//! Prepared bank reads use the same selected loans and numerical copy worker.
use super::*;
use crate::backend::{
    nn::workspace::ExistingArrayProjection,
    runtime::execution::generic::{
        LayerwiseWorkspace, MlxParameterPreparation, ParameterConstructors,
    },
    runtime::residency::manager::SelectedResidencySource,
    submission_recovery::native_role::physical,
};
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceContext, WorkspaceTensor},
};
use eredu_runtime::{LayerwiseAcquireError, working_memory::WorkingMemoryError};
use std::mem::{size_of, size_of_val};

fn unknown(stage: &'static str) -> Error {
    Error::OriginalSourceContract {
        stage,
        cause: WorkingMemoryError::UnknownBound,
    }
}
fn identity() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}
fn context(preparation: &MlxParameterPreparation<'_>) -> Result<WorkspaceContext, Error> {
    preparation
        .mechanism
        .context(preparation.funding.clone())
        .map_err(|cause| Error::Neural(cause.into()))
}
fn copied(
    source: &Array,
    preparation: &MlxParameterPreparation<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    let context = context(preparation)?;
    let completed = physical::copy_array(
        source,
        preparation.mechanism,
        &context,
        preparation.environment,
        preparation.execution,
        stream,
    )?;
    preparation.retain_source(completed.source, &context)?;
    Ok(completed.value)
}
fn joined(
    values: &[Array],
    preparation: &MlxParameterPreparation<'_>,
    stream: &Stream,
) -> Result<Array, Error> {
    if values.is_empty() {
        return Err(identity());
    }
    let context = context(preparation)?;
    let mut projection = ExistingArrayProjection::with_source_count(&context, values.len())
        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
    let mut symbolic = context.metadata_vec(values.len()).map_err(Error::Neural)?;
    let mut inputs = context.metadata_vec(values.len()).map_err(Error::Neural)?;
    for value in values {
        symbolic.push(projection.project(value).map_err(Error::Neural)?);
        inputs.push(value);
    }
    if !projection.is_complete() {
        return Err(unknown("parameter bank copied-member completed backing"));
    }
    context.begin_span();
    let output = WorkspaceTensor::concatenate(&symbolic, 0, &context).map_err(Error::Neural)?;
    context.complete_values(&[&output]).map_err(Error::Neural)?;
    let report = context.finish_report(&[output]).map_err(Error::Neural)?;
    let controls = [
        size_of::<(&[Array], &MlxParameterPreparation<'_>, &Stream)>(),
        size_of::<Array>(),
        size_of::<Result<Array, Error>>(),
        safemlx::ops::concatenate_axis_control_bytes().ok_or_else(overflow)?,
        crate::backend::runtime::cache::completed_borrow_control_bytes().ok_or_else(overflow)?,
    ];
    let controls = controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
        .ok_or_else(overflow)?;
    let completed = physical::execute_numerical(
        &report,
        &inputs,
        1,
        preparation.mechanism,
        &context,
        preparation.environment,
        preparation.execution,
        controls,
        || {
            let value = concatenate_axis(values, 0, stream)?;
            crate::backend::runtime::cache::complete_and_borrow(&value, stream)?;
            Ok(value)
        },
    )?;
    preparation.retain_source(completed.source, &context)?;
    Ok(completed.value)
}

impl SharedAddressableParameterBank {
    pub(super) fn with_prepared_parameter_slots(
        &self,
        bank: usize,
        unit: usize,
        parameters: &[PreparedBankParameter],
        operation: &mut ParameterSlotOperation<'_, MlxTensor, Error>,
        stream: &Stream,
        preparation: &MlxParameterPreparation<'_>,
    ) -> Result<bool, Error> {
        let context = context(preparation)?;
        let fixed = [
            size_of::<Self>(),
            size_of::<PreparedParameterLocation>(),
            size_of::<ResidencyManager>(),
            size_of::<SelectedResidencySource>(),
            size_of::<LayerwiseWorkspace>(),
            size_of::<ParameterConstructors>(),
            size_of::<Option<Array>>(),
            size_of::<Result<bool, Error>>(),
            size_of::<LayerwiseAcquireError<Error, Error>>(),
            size_of::<OffloadUnitId>(),
            size_of::<(
                &Self,
                usize,
                usize,
                &[PreparedBankParameter],
                &MlxParameterPreparation<'_>,
            )>(),
        ];
        context
            .charge_metadata(
                fixed
                    .into_iter()
                    .try_fold(size_of_val(&fixed), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(|cause| Error::Neural(cause.into()))?;
        let mut values = context
            .metadata_vec::<(&eredu_nn::ParameterMetadata, MlxTensor)>(
                preparation.selected_parameters().count(),
            )
            .map_err(Error::Neural)?;
        let location = PreparedParameterLocation::Bank { bank, unit };
        for id in preparation.selected_parameters() {
            if values
                .iter()
                .any(|(metadata, _)| metadata.id.as_str() == id)
            {
                continue;
            }
            let parameter = parameters
                .iter()
                .find(|p| p.slot.parameter.id.as_str() == id && p.slot.location == location)
                .ok_or_else(identity)?;
            let clone_bytes = safemlx::PreparedArrayClone::control_bytes()
                .and_then(|n| n.checked_add(Array::inspection_clone_handle_bytes()))
                .ok_or_else(overflow)?;
            context
                .charge_metadata(clone_bytes)
                .map_err(|cause| Error::Neural(cause.into()))?;
            let mut replacement_slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
            let replacement = self
                .with_workspace_source(preparation.funding, |loan| {
                    let member = loan
                        .members()
                        .find(|m| m.parameter == id)
                        .ok_or_else(identity)?;
                    loan.replacement(member)
                        .map(|(value, _)| {
                            replacement_slot
                                .fill_for_inspection(value.as_array())
                                .map_err(|cause| Error::Neural(context.metadata_source(cause)))
                        })
                        .transpose()
                })
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))??;
            drop(replacement_slot);
            let value = if let Some(value) = replacement {
                copied(&value, preparation, stream)?
            } else {
                let (manager, source, workspace) = self
                    .with_workspace_source(preparation.funding, |loan| {
                        let manager = loan.manager();
                        let source = manager
                            .supplementary_residency_source()
                            .ok_or_else(|| unknown("parameter bank supplementary residency"))?;
                        context
                            .charge_metadata(
                                source.projection_control_bytes().ok_or_else(overflow)?,
                            )
                            .map_err(|cause| Error::Neural(cause.into()))?;
                        Ok::<_, Error>((
                            manager.clone(),
                            SelectedResidencySource::Supplementary(source.clone()),
                            LayerwiseWorkspace::from_supplementary_source(
                                manager, source, &context,
                            )?,
                        ))
                    })
                    .map_err(|cause| Error::Neural(context.metadata_source(cause)))??;
                let mut copies = context
                    .metadata_vec(parameter.members.len())
                    .map_err(Error::Neural)?;
                for member in &parameter.members {
                    if member.parameter != id
                        || member.key.bank() != bank
                        || member.key.unit() != unit
                    {
                        return Err(identity());
                    }
                    context
                        .charge_metadata(member.key.unit_id_length())
                        .map_err(|cause| Error::Neural(cause.into()))?;
                    let unit_id = member.key.unit_id();
                    context
                        .charge_metadata(clone_bytes)
                        .map_err(|cause| Error::Neural(cause.into()))?;
                    let mut source_slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
                        .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
                    let outcome = preparation.inspect_source(
                        &manager,
                        &source,
                        &workspace,
                        &unit_id,
                        ParameterConstructors::default(),
                        |_| Ok::<Option<Array>, Error>(None),
                        |unit, lease, _| {
                            unit.inner = Some(
                                source_slot
                                    .fill_for_inspection(lease.device_value(&member.binding)?)
                                    .map_err(|cause| {
                                        Error::Neural(context.metadata_source(cause))
                                    })?,
                            );
                            Ok(())
                        },
                        |source| {
                            copies.push(copied(
                                source.as_ref().ok_or_else(identity)?,
                                preparation,
                                stream,
                            )?);
                            Ok(())
                        },
                        stream,
                    )?;
                    match outcome {
                        Ok(true) => (),
                        Ok(false) => return Err(identity()),
                        Err(
                            LayerwiseAcquireError::Architecture(cause)
                            | LayerwiseAcquireError::Policy(cause),
                        ) => return Err(cause),
                    }
                }
                joined(&copies, preparation, stream)?
            };
            values.push((&parameter.slot.parameter, MlxTensor::from_array(value)));
        }
        if values.is_empty() {
            return Err(identity());
        }
        operation(&mut |visitor| {
            for (metadata, value) in &mut values {
                visitor.visit_slot(metadata.as_view(), value);
            }
        })?;
        Ok(true)
    }
}
