//! Capability reports retained from the actual completed architecture handoff.

use super::{
    PreparedReplicatedTextArchitecture, ReplicatedTextConstructionSource,
    ReplicatedTextDispatchError,
};
use crate::capability::CapabilityEstimate;
use eredu_nn::{NeuralBackend, Tensor, workspace::WorkspaceMetadataError};
use eredu_runtime::SelectedReplicatedTextRealization;

pub(crate) fn prepare<B: NeuralBackend, E, F>(
    source: &impl ReplicatedTextConstructionSource,
    selected: &SelectedReplicatedTextRealization,
    context: &<B::Tensor as Tensor>::Context,
    construct: F,
) -> Result<CapabilityEstimate, ReplicatedTextDispatchError<E>>
where
    F: FnOnce() -> Result<CapabilityEstimate, eredu_core::CapabilityError>,
{
    let metadata = B::construction_metadata(context);
    if let Some(source) = source.prepared_sources() {
        if !std::ptr::eq(
            source.selected().text_realization().requirements(),
            selected.requirements(),
        ) {
            return Err(ReplicatedTextDispatchError::Metadata(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        if let Some(estimate) = source.construction_semantics().capability.get() {
            if let Some(metadata) = metadata {
                let controls = [
                    std::mem::size_of::<CapabilityEstimate>(),
                    std::mem::size_of::<Result<CapabilityEstimate, ReplicatedTextDispatchError<E>>>(
                    ),
                    std::mem::size_of::<F>(),
                ];
                let bytes = controls
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| {
                        ReplicatedTextDispatchError::Metadata(
                            WorkspaceMetadataError::Overflow.into(),
                        )
                    })?;
                metadata
                    .charge_metadata(bytes)
                    .map_err(|error| ReplicatedTextDispatchError::Metadata(error.into()))?;
            }
            return Ok(estimate.clone());
        }
    }
    if metadata.is_some_and(|context| context.uses_checked_metadata()) {
        // The cold report has not been constructed and retained yet. A checked
        // reconstruction cannot call the ordinary allocating report producer.
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    construct().map_err(|error| ReplicatedTextDispatchError::Architecture(error.to_string()))
}

pub(crate) fn publish<B: NeuralBackend, A, E>(
    source: &impl ReplicatedTextConstructionSource,
    prepared: &PreparedReplicatedTextArchitecture<A>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), ReplicatedTextDispatchError<E>> {
    // Workspace-only inspection must not create a completed model-construction
    // witness. Ordinary construction publishes after the actual contract succeeds.
    if B::construction_metadata(context).is_some() {
        return Ok(());
    }
    let Some(source) = source.prepared_sources() else {
        return Ok(());
    };
    if !std::ptr::eq(
        source.selected().text_realization().requirements(),
        prepared.selected().requirements(),
    ) {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    let retained = source
        .construction_semantics()
        .capability
        .get_or_init(|| prepared.capability_estimate().clone());
    if retained != prepared.capability_estimate() {
        return Err(ReplicatedTextDispatchError::Metadata(
            WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    Ok(())
}
