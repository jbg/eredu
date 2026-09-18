//! The same initial target source and completed construction facts for routed use.
use super::*;
use crate::{
    prepared_sources::PreparedModelSources,
    processor_plan::ArtifactArchitecturePlan,
    replicated_text::{ReplicatedTextConstructionSource, ReplicatedTextDispatchError},
};

/// Sealed architecture source retaining its exact admission/catalog inspection.
/// A prepared source lends the target graph and its completed semantic slots.
pub trait RoutedTextConstructionSource: ReplicatedTextConstructionSource {
    /// Exact target-only inspection, never rebuilt from caller metadata.
    fn routed_inspection(&self) -> &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>;
}
impl RoutedTextConstructionSource for eredu_core::ArtifactInspection<ArtifactArchitecturePlan> {
    fn routed_inspection(&self) -> &Self {
        self
    }
}
impl RoutedTextConstructionSource for PreparedModelSources {
    fn routed_inspection(&self) -> &eredu_core::ArtifactInspection<ArtifactArchitecturePlan> {
        self.execution_inspection()
    }
}
impl<T: RoutedTextConstructionSource + ?Sized> RoutedTextConstructionSource for &T {
    fn routed_inspection(&self) -> &eredu_core::ArtifactInspection<ArtifactArchitecturePlan> {
        T::routed_inspection(self)
    }
}
pub(super) fn preparation(
    error: ReplicatedTextDispatchError<std::convert::Infallible>,
) -> RoutedTextPreparationError {
    match error {
        ReplicatedTextDispatchError::Ineligible(_) => RoutedTextPreparationError::Ineligible,
        ReplicatedTextDispatchError::Architecture(message) => {
            RoutedTextPreparationError::Invalid(message)
        }
        ReplicatedTextDispatchError::Metadata(cause) => RoutedTextPreparationError::Metadata(cause),
        ReplicatedTextDispatchError::Backend(never) => match never {},
    }
}
/// Actual addressable target set used by the successful initial contract.
#[derive(Clone, Debug)]
pub(super) struct RetainedTargets(Option<std::sync::Arc<BTreeSet<String>>>);
impl Drop for RetainedTargets {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}
impl std::ops::Deref for RetainedTargets {
    type Target = BTreeSet<String>;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live routed addressable targets")
    }
}
/// Actual immutable ownership description emitted by the successful constructor.
#[derive(Clone, Debug)]
pub(crate) struct RetainedRoutedDescription(
    Option<std::sync::Arc<eredu_runtime::ArchitectureParameterDescription>>,
);
impl Drop for RetainedRoutedDescription {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(std::sync::Arc::into_inner(owner));
        }
    }
}
impl RetainedRoutedDescription {
    pub(crate) fn from_completed(description: eredu_runtime::ArchitectureParameterDescription) -> Self {
        Self(Some(std::sync::Arc::new(description)))
    }
}
impl std::ops::Deref for RetainedRoutedDescription {
    type Target = eredu_runtime::ArchitectureParameterDescription;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live routed parameter source")
    }
}
pub(crate) trait RoutedConstructionParameters<B: eredu_nn::NeuralBackend>:
    eredu_runtime::ArchitectureParameters<B, DefinitionError = eredu_nn::Error>
{
    fn install_construction_parameters(&mut self, source: RetainedRoutedDescription);
    fn prepare_construction_units(&self, _selected: Option<&RetainedRoutedBanks>, _context: &<B::Tensor as Tensor>::Context)
        -> Result<Option<RetainedRoutedUnits>, eredu_nn::Error> { Ok(None) }
    fn install_construction_units(&mut self, source: Option<RetainedRoutedUnits>) -> Result<(), eredu_nn::Error> {
        if source.is_some() { return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()); }
        Ok(())
    }

}
#[derive(Clone, Debug)]
pub(super) struct ParameterSources {
    target: RetainedRoutedDescription,
    source: Option<RetainedRoutedDescription>,
    target_units: Option<RetainedRoutedUnits>,
    source_units: Option<RetainedRoutedUnits>,
}
pub(super) fn parameters<B: eredu_nn::NeuralBackend, A: RoutedConstructionParameters<B>>(
    target: &mut A,
    source: Option<&mut A>,
    completed: Option<&CompletedRoutedConstruction>,
    banks: &RetainedRoutedBanks,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ParameterSources, RoutedTextPreparationError> {
    crate::decoder::ModuleMetadata::new::<B>(context)
        .controls::<(ParameterSources, Option<RetainedRoutedDescription>, &mut A)>()
        .map_err(RoutedTextPreparationError::Metadata)?;
    let result = match completed {
        Some(completed) => {
            if completed.parameters.source.is_some() != source.is_some() {
                return Err(RoutedTextPreparationError::Metadata(
                    eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
                ));
            }
            completed.parameters.clone()
        }
        None => {
            let target_units = target.prepare_construction_units(Some(banks), context)
                .map_err(RoutedTextPreparationError::Metadata)?;
            let source_units = source.as_ref().map(|source| source.prepare_construction_units(None, context))
                .transpose().map_err(RoutedTextPreparationError::Metadata)?.flatten();
            let target = target.parameter_description(context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(context))).map_err(|cause| {
                preparation(config_source::constructor_error::<
                    B,
                    std::convert::Infallible,
                >(cause, context))
            })?;
            let source = source
                .as_ref()
                .map(|source| source.parameter_description(context).and_then(|description| eredu_runtime::ArchitectureParameterDescription::into_owned(description,B::construction_metadata(context))))
                .transpose()
                .map_err(|cause| {
                    preparation(config_source::constructor_error::<
                        B,
                        std::convert::Infallible,
                    >(cause, context))
                })?;
            ParameterSources {
                target_units, source_units,
                target: RetainedRoutedDescription(Some(std::sync::Arc::new(target))),
                source: source
                    .map(|source| RetainedRoutedDescription(Some(std::sync::Arc::new(source)))),
            }
        }
    };
    target.install_construction_units(result.target_units.clone())
        .map_err(RoutedTextPreparationError::Metadata)?;
    target.install_construction_parameters(result.target.clone());
    if let (Some(source), Some(parameters)) = (source, result.source.as_ref()) {
        source.install_construction_units(result.source_units.clone())
            .map_err(RoutedTextPreparationError::Metadata)?;
        source.install_construction_parameters(parameters.clone());
    }
    Ok(result)
}
#[derive(Clone, Debug)]
pub(crate) struct CompletedRoutedConstruction {
    banks: RetainedRoutedBanks,
    targets: RetainedTargets,
    parameters: ParameterSources,
    materialization: eredu_runtime::PreparedContractMaterialization,
}
pub(super) fn begin<B: eredu_nn::NeuralBackend>(
    source: &impl RoutedTextConstructionSource,
    selected: &SelectedRoutedTextRealization,
    store: &eredu_checkpoint::store::RetainedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<CompletedRoutedConstruction>, RoutedTextPreparationError> {
    let metadata = B::construction_metadata(context);
    if let Some(metadata) = metadata {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<CompletedRoutedConstruction>(),
            size_of::<Option<CompletedRoutedConstruction>>(),
            size_of::<RetainedRoutedBanks>(),
            size_of::<RetainedTargets>(),
            size_of::<Option<&PreparedModelSources>>(),
            size_of::<Result<Option<CompletedRoutedConstruction>, RoutedTextPreparationError>>(),
            size_of::<(
                &SelectedRoutedTextRealization,
                &eredu_checkpoint::store::RetainedCheckpointSource,
            )>(),
            size_of::<Result<(), RoutedTextPreparationError>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or_else(|| {
                RoutedTextPreparationError::Metadata(
                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                )
            })?;
        metadata
            .charge_metadata(bytes)
            .map_err(|cause| RoutedTextPreparationError::Metadata(cause.into()))?;
    }
    crate::replicated_text::store_handoff::validate::<B, std::convert::Infallible>(
        source,
        selected.text(),
        store,
        context,
    )
    .map_err(preparation)?;
    if let Some(prepared) = source.prepared_sources() {
        if let Some(completed) = prepared.construction_semantics().routed.get() {
            if !completed.banks.same_source(&selected.banks) {
                return Err(RoutedTextPreparationError::Metadata(
                    eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
                ));
            }
            return Ok(Some(completed.clone()));
        }
    }
    if metadata.is_some_and(|metadata| metadata.uses_checked_metadata()) {
        return Err(RoutedTextPreparationError::Metadata(
            eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
        ));
    }
    Ok(None)
}
pub(super) fn targets(
    completed: &Option<CompletedRoutedConstruction>,
    ordinary: impl FnOnce() -> BTreeSet<String>,
) -> RetainedTargets {
    match completed {
        Some(completed) => completed.targets.clone(),
        None => RetainedTargets(Some(std::sync::Arc::new(ordinary()))),
    }
}
pub(super) fn publish<B: eredu_nn::NeuralBackend>(
    source: &impl RoutedTextConstructionSource,
    text: &eredu_runtime::SelectedReplicatedTextRealization,
    banks: &RetainedRoutedBanks,
    targets: &RetainedTargets,
    parameters: &ParameterSources,
    materialization: Option<&eredu_runtime::PreparedContractMaterialization>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(), RoutedTextPreparationError> {
    if B::construction_metadata(context).is_some() {
        return Ok(());
    }
    if let Some(source) = source.prepared_sources() {
        if !config_source::exact_selection(source, text) {
            return Err(RoutedTextPreparationError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        let materialization = materialization.ok_or_else(|| {
            RoutedTextPreparationError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
            )
        })?;
        let completed =
            source
                .construction_semantics()
                .routed
                .get_or_init(|| CompletedRoutedConstruction {
                    banks: banks.clone(),
                    targets: targets.clone(),
                    parameters: parameters.clone(),
                    materialization: materialization.clone(),
                });
        if !completed.banks.same_source(banks) {
            return Err(RoutedTextPreparationError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn materialization(
    completed: &Option<CompletedRoutedConstruction>,
) -> Option<&eredu_runtime::PreparedContractMaterialization> {
    completed
        .as_ref()
        .map(|completed| &completed.materialization)
}
pub(super) fn contract(
    cause: eredu_runtime::PreparedTextContractError,
) -> RoutedTextPreparationError {
    match cause {
        eredu_runtime::PreparedTextContractError::Contract(message) => {
            RoutedTextPreparationError::Invalid(message)
        }
        eredu_runtime::PreparedTextContractError::Metadata(cause) => {
            RoutedTextPreparationError::Metadata(cause)
        }
    }
}
