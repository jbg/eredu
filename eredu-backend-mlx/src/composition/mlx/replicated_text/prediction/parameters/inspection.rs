//! Read-only parameter traversal under the actual supplementary source loan.
use super::*;
use crate::backend::runtime::execution::generic::{
    LayerwiseWorkspace, MlxParameterPreparation, ParameterConstructors,
};
use crate::backend::runtime::residency::manager::SelectedResidencySource;
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

fn identity(stage: &'static str) -> Error {
    Error::OriginalSourceContract {
        stage,
        cause: WorkingMemoryError::IdentityMismatch,
    }
}
fn overflow() -> Error {
    Error::PrefillControl(WorkingMemoryError::Overflow)
}

pub(super) fn with_slots<M: Parameterized<MlxTensor>>(
    module: &MlxPredictionModule<M>,
    operation: &mut ParameterSlotOperation<'_, MlxTensor, Error>,
    preparation: &MlxParameterPreparation<'_>,
) -> Result<(), Error> {
    let context = preparation
        .mechanism
        .context(preparation.funding.clone())
        .map_err(|cause| Error::Neural(cause.into()))?;
    let frames = [
        size_of::<Vec<MlxTensor>>(),
        size_of::<Vec<safemlx::PreparedArrayClone>>(),
        size_of::<SelectedResidencySource>(),
        size_of::<Result<bool, eredu_runtime::LayerwiseAcquireError<Infallible, Error>>>(),
        size_of::<(&MlxPredictionModule<M>, &MlxParameterPreparation<'_>)>(),
        WorkspaceContext::metadata_source_bytes::<safemlx::PreparedArrayCloneCause>()
            .ok_or_else(overflow)?,
    ];
    let handle_bytes = Array::inspection_clone_handle_bytes()
        .checked_add(safemlx::PreparedArrayClone::control_bytes().ok_or_else(overflow)?)
        .and_then(|bytes| bytes.checked_mul(module.parameters.len()))
        .ok_or_else(overflow)?;
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .and_then(|bytes| bytes.checked_add(handle_bytes))
                .ok_or_else(overflow)?,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
    let manager = module
        .manager
        .get()
        .ok_or_else(|| identity("supplementary parameter source manager"))?;
    let id = module
        .id
        .as_ref()
        .ok_or_else(|| identity("supplementary parameter source unit"))?;
    let source = manager
        .supplementary_residency_source()
        .ok_or_else(|| identity("supplementary parameter source inventory"))?;
    let workspace = LayerwiseWorkspace::from_supplementary_source(manager, source, &context)?;
    let selected = SelectedResidencySource::Supplementary(source.clone());
    let mut handles = context
        .metadata_vec(module.parameters.len())
        .map_err(Error::Neural)?;
    for _ in &module.parameters {
        handles.push(
            safemlx::PreparedArrayClone::try_prepare_for_inspection()
                .map_err(|cause| Error::Neural(context.metadata_source(cause)))?,
        );
    }
    let values = context
        .metadata_vec(module.parameters.len())
        .map_err(Error::Neural)?;
    // The exact prepared slot identities address the same manager bindings as
    // module population. Read-only visitors need no module mutation or cache reset.
    let outcome = preparation.inspect_source(
        manager,
        &selected,
        &workspace,
        id,
        ParameterConstructors::default(),
        |_| Ok::<_, Infallible>(values),
        |values, lease, _| {
            for (slot, handle) in module.parameters.iter().zip(&mut handles) {
                let value = match module.replacements.get(slot.parameter.id.as_str()) {
                    Some(value) => value.as_array(),
                    None => lease.device_value(slot.parameter.id.as_str())?,
                };
                let value = handle
                    .fill_for_inspection(value)
                    .map_err(|cause| Error::Neural(context.metadata_source(cause)))?;
                values.push(MlxTensor::from_array(value));
            }
            Ok(())
        },
        |values| {
            operation(&mut |visitor| {
                for (slot, value) in module.parameters.iter().zip(values.iter()) {
                    visitor.visit_slot(slot.parameter.as_view(), value);
                }
            })
        },
        &module.stream,
    );
    let outcome = outcome.and_then(|outcome| match outcome {
        Ok(true) => Ok(()),
        Ok(false) => Err(identity("supplementary parameter source callback")),
        Err(eredu_runtime::LayerwiseAcquireError::Architecture(never)) => match never {},
        Err(eredu_runtime::LayerwiseAcquireError::Policy(cause)) => Err(cause),
    });
    if !module.residency.is_fully_resident() {
        let eviction = manager.evict(id, MemoryTier::Device);
        if outcome.is_ok() {
            eviction?;
        }
    }
    outcome
}
