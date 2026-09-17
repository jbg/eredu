//! Composite contracts consume the same completed routed selection and materialization.
use super::*;
use crate::{
    prepared_sources::PreparedModelSources, replicated_text::PreparedReplicatedTextArchitecture,
};
use eredu_nn::{Error, workspace::WorkspaceMetadataError};
use eredu_runtime::PreparedTextContractError;

fn contract(cause: RoutedTextPreparationError) -> PreparedTextContractError {
    match cause {
        RoutedTextPreparationError::Metadata(cause) => cause.into(),
        RoutedTextPreparationError::Invalid(message) => {
            PreparedTextContractError::Contract(message)
        }
        RoutedTextPreparationError::Ineligible => {
            Error::from(WorkspaceMetadataError::Unqualified).into()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_composite_handoff<B, S, A>(
    mut architecture: A,
    mut source_architecture: Option<A>,
    original: Option<&PreparedModelSources>,
    expected: &RoutedTextRequirements,
    selected: SelectedRoutedTextRealization,
    store: &eredu_checkpoint::store::RetainedCheckpointSource,
    capability: crate::capability::CapabilityEstimate,
    model_type: String,
    identity: String,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedRoutedTextArchitecture<A>, PreparedTextContractError>
where
    B: GroupedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B>,
    A: eredu_runtime::LayeredArchitecture<B, S, Error = Error>
        + eredu_runtime::RoutedLayeredArchitecture<B, S>
        + source::RoutedConstructionParameters<B>,
    A::StaticModules: Clone,
{
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    metadata.controls::<(
        A,
        Option<A>,
        Option<&PreparedModelSources>,
        &RoutedTextRequirements,
        SelectedRoutedTextRealization,
        &eredu_checkpoint::store::RetainedCheckpointSource,
        crate::capability::CapabilityEstimate,
        String,
        String,
        Option<source::CompletedRoutedConstruction>,
        source::RetainedTargets,
        source::ParameterSources,
        (&eredu_runtime::ParameterBankResidency, &RetainedRoutedBanks),
        PreparedReplicatedTextArchitecture<A>,
        PreparedRoutedTextArchitecture<A>,
    )>()?;
    // A prepared source carries the actual store/selection identity and its bank
    // owner. Checked callers cannot build a new source from equal declarations.
    let completed = match original {
        Some(original) => {
            source::begin::<B>(original, &selected, store, context).map_err(contract)?
        }
        None if metadata.context().is_some() => {
            return Err(Error::from(WorkspaceMetadataError::Unqualified).into());
        }
        None => None,
    };
    if completed.is_none() {
        validate_selected_routed_handoff(expected, &selected)
            .map_err(|cause| PreparedTextContractError::Contract(cause.to_string()))?;
    }
    let (text, bank_residency, banks) = selected.into_shared_parts();
    let targets = source::targets(&completed, || {
        addressable_bank_parameter_targets(&bank_residency, &banks)
    });
    let parameters = source::parameters::<B, A>(
        &mut architecture,
        source_architecture.as_mut(),
        completed.as_ref(),
        &banks,
        context,
    )
    .map_err(contract)?;
    let prepared =
        crate::replicated_text::prepare_architecture_handoff_with_addressable_metadata::<B, S, A>(
            architecture,
            source_architecture,
            text,
            capability,
            model_type,
            identity,
            targets.iter().map(String::as_str),
            source::materialization(&completed),
            context,
        )?;
    if let Some(original) = original {
        source::publish::<B>(
            original,
            prepared.selected(),
            &banks,
            &targets,
            &parameters,
            prepared.materialization_source(),
            context,
        )
        .map_err(contract)?;
    }
    Ok(PreparedRoutedTextArchitecture {
        text: prepared,
        bank_residency,
        banks,
    })
}
