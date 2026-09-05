//! MLX checkpoint materialization after backend-neutral planning.

use std::sync::Arc;

use eredu_architectures::{
    prepared_sources::{prepare_model_sources, PreparedModelSourceGraph, PreparedModelSources},
    processor_plan::ArtifactArchitecturePlan,
};
use safemlx::Stream;

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;

use crate::{
    backend::error::Error,
    backend::MlxModel,
    composition::{mlx::Executable, MlxNeuralBackend},
    MlxLoadRequest,
};

/// Opaque MLX model configuration selected before payloads are opened.
pub struct MlxModelConfig {
    pub(crate) sources: PreparedModelSources,
    pub(crate) rank_context: Option<crate::backend::MlxRankContext>,
}

impl MlxModelConfig {
    pub(crate) fn new(
        selected: eredu_core::SelectedModelPreparation<crate::backend::MlxBackend<'_>>,
    ) -> Result<Self, Error> {
        let (plan, selected) = selected.into_parts();
        let (sources, rank_context) = prepare_selected_sources(plan, selected)?;
        Ok(Self {
            sources,
            rank_context,
        })
    }
}

pub(crate) fn prepare_selected_sources(
    plan: eredu_core::ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: MlxSelectedPreparation,
) -> Result<(PreparedModelSources, Option<crate::backend::MlxRankContext>), Error> {
    let MlxSelectedPreparation {
        selected,
        rank_context,
    } = selected;
    let sources = prepare_model_sources(plan, selected)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    #[cfg(test)]
    if sources.format() == eredu_core::ArtifactFormat::Gguf {
        super::path_instrumentation::payload_open();
        for _ in sources.companions() {
            super::path_instrumentation::payload_open();
        }
    }
    Ok((sources, rank_context))
}

/// Opaque, authoritative MLX construction policy selected before payloads are opened.
#[derive(Debug, Clone)]
pub struct MlxSelectedPreparation {
    selected: eredu_architectures::SelectedPreparation,
    rank_context: Option<crate::backend::MlxRankContext>,
}

impl MlxSelectedPreparation {
    const fn new(
        selected: eredu_architectures::SelectedPreparation,
        rank_context: Option<crate::backend::MlxRankContext>,
    ) -> Self {
        Self {
            selected,
            rank_context,
        }
    }

    pub(crate) const fn session_capabilities(&self) -> eredu_core::SessionCapabilities {
        self.selected.session_capabilities()
    }

    pub(crate) const fn neutral(&self) -> &eredu_architectures::SelectedPreparation {
        &self.selected
    }

    #[cfg(test)]
    pub(crate) const fn rank_context(&self) -> Option<crate::backend::MlxRankContext> {
        self.rank_context
    }
}

pub(crate) fn select_preparation(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(&super::replicated_text::GROUPED_OPERATION_CAPABILITIES),
    )
}

pub(crate) struct MlxPreparationMechanisms<'a> {
    grouped: &'a [eredu_runtime::GroupedOperationRequirement],
    communication: Option<&'a eredu_runtime::CommunicationCapabilities>,
}

impl<'a> MlxPreparationMechanisms<'a> {
    pub(crate) const fn new(grouped: &'a [eredu_runtime::GroupedOperationRequirement]) -> Self {
        Self {
            grouped,
            communication: None,
        }
    }

    #[cfg(test)]
    const fn with_communication(
        mut self,
        communication: &'a eredu_runtime::CommunicationCapabilities,
    ) -> Self {
        self.communication = Some(communication);
        self
    }
}

impl eredu_architectures::PreparationMechanismProvider for MlxPreparationMechanisms<'_> {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::structural::preparation_mechanism_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        self.grouped.contains(&requirement)
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        super::replicated_text::capabilities(requirements, request)
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        super::processor::capabilities()
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        super::speculative::speculative_mechanism_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        self.communication.cloned().unwrap_or_else(|| {
            crate::backend::runtime::distributed::topology::mlx_communication_capabilities()
        })
    }
}

#[cfg(test)]
pub(crate) fn select_preparation_with_grouped_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(grouped_capabilities),
    )
}

#[cfg(test)]
fn select_preparation_with_mechanism_capabilities(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    grouped_capabilities: &[eredu_runtime::GroupedOperationRequirement],
    communication: &eredu_runtime::CommunicationCapabilities,
) -> Result<MlxSelectedPreparation, Error> {
    select_preparation_with_mechanisms(
        inspection,
        options,
        &MlxPreparationMechanisms::new(grouped_capabilities).with_communication(communication),
    )
}

fn select_preparation_with_mechanisms(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    options: MlxLoadRequest,
    mechanisms: &impl eredu_architectures::PreparationMechanismProvider,
) -> Result<MlxSelectedPreparation, Error> {
    let (request, rank_context) = options.checked_normalized()?;
    let selected = eredu_architectures::preparation_selection::select_preparation(
        inspection, request, mechanisms,
    )
    .map_err(preparation_selection_error)?;
    Ok(MlxSelectedPreparation::new(selected, rank_context))
}

fn preparation_selection_error(error: eredu_architectures::PreparationSelectionError) -> Error {
    match error {
        eredu_architectures::PreparationSelectionError::Admission(error) => {
            Error::PreparationAdmission(error)
        }
        error => Error::ArchitectureModel(error.to_string()),
    }
}

pub(crate) fn materialize_model_plan(
    sources: PreparedModelSources,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let (selected, inspection, sources) = sources.into_parts();
    let (execution, _, _, prediction_realization) = selected.into_parts();
    let materializer = MlxSelectedExecutionMaterializer {
        inspection,
        sources,
        prediction_realization,
        distributed,
        stream,
        weights_stream,
    };
    execution.dispatch(materializer)
}

fn materialize_partitioned_composite(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::replicated_text::SelectedCompositeTextRealization,
        eredu_architectures::replicated_text::CompositeTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral composite binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().execution().state().policy().clone();
    let selected_processor = selected.base().processor().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected
                    .base()
                    .execution()
                    .auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::composite_partitioned::visit_authoritative_composite_prediction_target_partition::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                selected,
                execution,
                stream,
                super::replicated_text::PartitionedCompositePredictionBindingVisitor {
                    store: target_store,
                    distributed,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::replicated_text::bind_partitioned_composite(
            selected,
            target_store,
            distributed,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed);
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

fn materialize_partitioned_routed_decoder(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_architectures::SelectedRoutedTextRealization,
        eredu_architectures::RoutedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral routed-decoder binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().text().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let prediction_extension_sources = sources
        .extension()
        .map(|source| source.source_keys().into_iter().collect())
        .unwrap_or_default();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected
                    .base()
                    .text()
                    .auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::bind_partitioned_routed_prediction_decoder(
                &inspection,
                selected,
                execution,
                realization,
                capability,
                target_store,
                distributed,
                prediction_extension_sources,
                stream,
                weights_stream,
            )?
        }
        (None, None, None) => super::replicated_text::bind_partitioned_routed_decoder(
            &inspection,
            selected,
            target_store,
            distributed,
            prediction_extension_sources,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    Ok(MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed))
}

fn materialize_partitioned_dense_decoder(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    selected: eredu_architectures::partitioned_execution::SelectedPartitionedAdmission<
        eredu_runtime::SelectedReplicatedTextRealization,
        eredu_runtime::ReplicatedTextRequirements,
    >,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let distributed = distributed.ok_or_else(|| {
        Error::Parallel("neutral dense-decoder binding has no realized communication".into())
    })?;
    let retained_distributed = distributed.clone();
    let state_residency = selected.base().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let prediction_extension_sources = sources
        .extension()
        .map(|source| source.source_keys().into_iter().collect())
        .unwrap_or_default();
    let target_store = Arc::clone(sources.target());
    let extension_store = sources.extension().cloned();
    let prediction_extension_execution = match (prediction_extension.as_ref(), extension_store) {
        (Some(extension), Some(extension_store)) => {
            let prepared = eredu_architectures::prediction_extension::prepare_partitioned_prediction_extension::<
                MlxNeuralBackend,
                _,
                _,
            >(
                extension,
                &selected,
                selected.base().auxiliary_materialization_tasks(),
                weights_stream,
                stream,
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            super::replicated_text::materialize_prediction_extension(
                prepared,
                extension_store.as_ref(),
                stream,
                weights_stream,
            )
            .map(Some)
            .map_err(|error| {
                Error::ArchitectureModel(format!(
                    "prediction extension materialization failed: {error}"
                ))
            })?
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let executable = match (
        prediction_extension,
        prediction_extension_execution,
        prediction_realization,
    ) {
        (Some(extension), Some(execution), Some(realization)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::partitioned_execution::visit_resident_partitioned_prediction_target_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                &inspection,
                selected,
                execution,
                target_store,
                stream,
                super::replicated_text::PartitionedPredictionBindingVisitor {
                    distributed,
                    additional_claimed_sources: prediction_extension_sources,
                    stream,
                    weights_stream,
                    selected: realization,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::replicated_text::bind_partitioned_dense_decoder(
            &inspection,
            selected,
            target_store,
            distributed,
            prediction_extension_sources,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    Ok(MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    )
    .with_distributed(retained_distributed))
}

struct MlxSelectedExecutionMaterializer<'a> {
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    distributed: Option<crate::backend::distributed::MlxDistributedSession>,
    stream: &'a Stream,
    weights_stream: &'a Stream,
}

impl eredu_architectures::SelectedExecutionDispatcher for MlxSelectedExecutionMaterializer<'_> {
    type Output = MlxModel;
    type Error = Error;

    fn replicated(
        self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_replicated_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn routed(
        self,
        selected: eredu_architectures::SelectedRoutedTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_routed_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn composite(
        self,
        selected: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<Self::Output, Self::Error> {
        debug_assert!(self.distributed.is_none());
        materialize_composite_text_plan(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_dense(
        self,
        selected: eredu_architectures::SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_dense_decoder(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_routed(
        self,
        selected: eredu_architectures::SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_routed_decoder(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }

    fn partitioned_composite(
        self,
        selected: eredu_architectures::SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, Self::Error> {
        materialize_partitioned_composite(
            self.inspection,
            self.sources,
            self.prediction_realization,
            selected,
            self.distributed,
            self.stream,
            self.weights_stream,
        )
    }
}

fn materialize_replicated_prediction_extension(
    extension: &eredu_architectures::configuration::PredictionExtensionPlan,
    tasks: &[eredu_runtime::ReplicatedTextMaterializationTask],
    store: &dyn eredu_checkpoint::store::CheckpointSource,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<super::replicated_text::MaterializedEmbeddedPrediction, Error> {
    let prepared =
        eredu_architectures::prediction_extension::prepare_replicated_prediction_extension::<
            MlxNeuralBackend,
        >(extension, tasks, weights_stream, stream)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    super::replicated_text::materialize_prediction_extension(
        prepared,
        store,
        stream,
        weights_stream,
    )
}

fn materialize_replicated_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_runtime::SelectedReplicatedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::replicated_text::dispatch_replicated_prediction_target_architecture(
                &architecture_plan,
                realization,
                materialized,
                target_store,
                stream,
                super::replicated_text::PredictionBindingVisitor {
                    stream,
                    weights_stream,
                    selected,
                    capability,
                },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => bind_replicated_text(
            &architecture_plan,
            realization,
            target_store,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_routed_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_architectures::SelectedRoutedTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.text().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.text().auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::routed_text::dispatch_routed_prediction_target_architecture(
                &inspection,
                realization,
                materialized,
                target_store,
                stream,
                super::replicated_text::PredictionBindingVisitor {
                    stream,
                    weights_stream,
                    selected,
                    capability,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => super::replicated_text::bind_routed_text(
            &inspection,
            realization,
            target_store,
            stream,
            weights_stream,
        )?,
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    Ok(model)
}

fn materialize_composite_text_plan(
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    sources: PreparedModelSourceGraph,
    prediction_realization: Option<eredu_runtime::SelectedSpeculativeRealization>,
    realization: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MlxModel, Error> {
    let state_residency = realization.execution().state().policy().clone();
    let architecture_plan = sources.architecture().clone();
    let inspection = inspection.map_architecture_plan(|_complete| architecture_plan.clone());
    let floating_state_dtype_bytes = inspected_floating_state_dtype_bytes(&inspection)?;
    let requirements =
        eredu_architectures::replicated_text::composite_text_requirements(&inspection)
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    let selected_processor = realization.processor().clone();
    let prediction_extension = sources.prediction_extension().cloned();
    let materialized = match (prediction_extension.as_ref(), sources.extension()) {
        (Some(extension), Some(extension_store)) => {
            Some(materialize_replicated_prediction_extension(
                extension,
                realization.execution().auxiliary_materialization_tasks(),
                extension_store.as_ref(),
                stream,
                weights_stream,
            )?)
        }
        (None, None) => None,
        _ => {
            return Err(Error::ArchitectureModel(
                "prepared prediction extension and source role disagree".into(),
            ))
        }
    };
    let target_store = Arc::clone(sources.target());
    let executable = match (prediction_extension, materialized, prediction_realization) {
        (Some(extension), Some(materialized), Some(selected)) => {
            let capability =
                eredu_architectures::prediction_extension::prediction_extension_capability(
                    &extension,
                )
                .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
            eredu_architectures::replicated_text::visit_composite_prediction_target_text_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                super::replicated_text::MlxEmbeddedPredictionMaterializer,
                _,
            >(
                requirements,
                realization,
                materialized,
                target_store,
                stream,
                super::replicated_text::PredictionBindingVisitor { stream, weights_stream, selected, capability },
            ).map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        (None, None, None) => {
            eredu_architectures::replicated_text::visit_composite_text_architecture::<
                MlxNeuralBackend,
                crate::backend::runtime::cache::state::MlxHybridState,
                _,
            >(
                requirements,
                realization,
                target_store,
                stream,
                super::replicated_text::CompositeBindingVisitor {
                    stream,
                    weights_stream,
                },
            )
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        }
        _ => {
            return Err(Error::ArchitectureModel(
                "embedded prediction selection and materialization disagree".into(),
            ))
        }
    };
    let model = MlxModel::new(
        Executable::new(executable),
        floating_state_dtype_bytes,
        state_residency,
    );
    attach_selected_processor(model, &architecture_plan, &selected_processor)
}

pub(super) fn bind_replicated_text(
    architecture_plan: &ArtifactArchitecturePlan,
    selected: eredu_runtime::SelectedReplicatedTextRealization,
    store: Arc<dyn eredu_checkpoint::store::CheckpointSource>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>, Error> {
    let visitor = super::replicated_text::BindingVisitor {
        stream,
        weights_stream,
    };
    eredu_architectures::replicated_text::dispatch_replicated_text_architecture::<
        crate::backend::nn::shared::MlxNeuralBackend,
        _,
    >(architecture_plan, selected, store, stream, visitor)
    .map_err(|error| Error::ArchitectureModel(error.to_string()))
}

fn inspected_floating_state_dtype_bytes(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
) -> Result<std::num::NonZeroU8, Error> {
    let source = eredu_architectures::preparation::prepared_floating_state_dtype_source(inspection)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    mlx_floating_state_dtype_bytes(source.dtype()).map_err(|dtype| {
        Error::ArchitectureModel(format!(
            "floating-state dtype source {:?} has unsupported MLX activation dtype {dtype:?}",
            source.checkpoint_tensor()
        ))
    })
}

fn mlx_floating_state_dtype_bytes(
    dtype: &eredu_core::checkpoint::TensorDtype,
) -> Result<std::num::NonZeroU8, eredu_core::checkpoint::TensorDtype> {
    use eredu_core::checkpoint::TensorDtype;

    let bytes = match dtype {
        TensorDtype::F16 | TensorDtype::Bf16 => 2,
        TensorDtype::F32 => 4,
        TensorDtype::F64 | TensorDtype::Complex64 => 8,
        // MLX materializes supported packed embeddings as Float32 activations.
        // These cases are reached only after the architecture schema resolved
        // the exact embedding parameter; they are not a fallback for an
        // unknown checkpoint name.
        TensorDtype::U32 | TensorDtype::Encoded(_) => 4,
        dtype => return Err(dtype.clone()),
    };
    Ok(std::num::NonZeroU8::new(bytes).expect("supported MLX activation widths are nonzero"))
}

#[cfg(test)]
#[allow(
    clippy::items_after_test_module,
    reason = "floating-state dtype tests stay adjacent to dtype resolution"
)]
mod floating_state_dtype_tests {
    use super::{
        inspected_floating_state_dtype_bytes, mlx_floating_state_dtype_bytes,
        select_preparation_with_mechanism_capabilities,
    };
    use crate::backend::{ExecutionContext, MlxBackend};
    use eredu_core::{
        checkpoint::TensorDtype, residency::OffloadConfig, ModelLoadingBackend as _,
        ParallelTopology,
    };
    use eredu_gguf::{GgmlType, MetadataValue, TensorInput, Writer};

    #[test]
    fn deepseek_v4_prediction_selection_retains_neutral_extension_contract() {
        let root = tempfile::tempdir().unwrap();
        crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(root.path(), 1);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            4,
            4096,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();

        let selected = super::select_preparation(&inspection, options).unwrap();

        let (target, extension) = inspection
            .architecture_plan()
            .prediction_target_projection()
            .unwrap()
            .unwrap();
        let extension_sources = extension
            .source_keys(target.safetensors_architecture().unwrap())
            .unwrap();
        assert_eq!(extension_sources.len(), 41);
        let target_sources = target
            .safetensors_architecture()
            .unwrap()
            .checkpoint_resolution()
            .unwrap()
            .source_keys();
        let complete_sources = extension
            .complete_architecture()
            .checkpoint_resolution()
            .unwrap()
            .source_keys();
        assert!(target_sources.is_disjoint(&extension_sources));
        assert_eq!(
            target_sources
                .union(&extension_sources)
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            *complete_sources
        );
        assert!(selected.neutral().prediction_extension().is_some());
        assert!(selected.neutral().communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
    }

    #[test]
    fn disabled_plan_projects_one_ordinary_target_without_selecting_extension_payloads() {
        let root = tempfile::tempdir().unwrap();
        crate::tests::distributed_pipeline_ring::write_deepseek_v4_fixture(root.path(), 1);
        let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
        let options = crate::MlxLoadRequest::default()
            .with_drafting_plan(&eredu_core::DraftingPlan::Disabled)
            .unwrap();

        let selected = super::select_preparation(&inspection, options).unwrap();

        assert!(selected.neutral().prediction_extension().is_none());
        assert!(selected.neutral().communication_manifest().is_none());
    }
    use safemlx::{Device, DeviceType};
    use std::collections::BTreeMap;

    fn write_minimal_llama_gguf(path: &std::path::Path, dtype: GgmlType) {
        let metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count".into(), MetadataValue::Uint32(2)),
            ("llama.embedding_length".into(), MetadataValue::Uint32(16)),
            (
                "llama.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "llama.feed_forward_length".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-5),
            ),
            ("llama.vocab_size".into(), MetadataValue::Uint32(16)),
            ("llama.context_length".into(), MetadataValue::Uint32(1)),
        ]);
        let vector_data = [0_u8; 32];
        let matrix_data = [0_u8; 512];
        let tensor = |name, dimensions| TensorInput {
            name,
            dimensions,
            ggml_type: dtype,
            data: if dimensions.len() == 1 {
                &vector_data
            } else {
                &matrix_data
            },
        };
        let tensors = [
            tensor("token_embd.weight", &[16, 16]),
            tensor("output_norm.weight", &[16]),
            tensor("blk.0.attn_norm.weight", &[16]),
            tensor("blk.0.ffn_norm.weight", &[16]),
            tensor("blk.0.attn_q.weight", &[16, 16]),
            tensor("blk.0.attn_k.weight", &[16, 16]),
            tensor("blk.0.attn_v.weight", &[16, 16]),
            tensor("blk.0.attn_output.weight", &[16, 16]),
            tensor("blk.0.ffn_gate.weight", &[16, 16]),
            tensor("blk.0.ffn_up.weight", &[16, 16]),
            tensor("blk.0.ffn_down.weight", &[16, 16]),
            tensor("blk.1.attn_norm.weight", &[16]),
            tensor("blk.1.ffn_norm.weight", &[16]),
            tensor("blk.1.attn_q.weight", &[16, 16]),
            tensor("blk.1.attn_k.weight", &[16, 16]),
            tensor("blk.1.attn_v.weight", &[16, 16]),
            tensor("blk.1.attn_output.weight", &[16, 16]),
            tensor("blk.1.ffn_gate.weight", &[16, 16]),
            tensor("blk.1.ffn_up.weight", &[16, 16]),
            tensor("blk.1.ffn_down.weight", &[16, 16]),
        ];
        Writer::default()
            .write(std::fs::File::create(path).unwrap(), &metadata, &tensors)
            .unwrap();
    }

    #[test]
    fn partitioned_capability_failure_precedes_native_or_payload_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();
        super::super::path_instrumentation::reset();

        let error = select_preparation_with_mechanism_capabilities(
            &inspection,
            options,
            &super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES,
            &eredu_runtime::CommunicationCapabilities::new([]).unwrap(),
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("communication"),
            "unexpected selection failure: {error}"
        );
        assert_eq!(
            super::super::path_instrumentation::communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            Default::default()
        );
    }

    #[test]
    fn data_parallel_request_reaches_neutral_selection_and_fails_before_native_or_payload_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology =
            eredu_core::ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 2).unwrap(), 0)
                .unwrap();
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();
        super::super::path_instrumentation::reset();

        let error = super::select_preparation(&inspection, options).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("data-parallel execution is not supported"),
            "unexpected data-parallel rejection: {error}"
        );
        assert_eq!(
            super::super::path_instrumentation::communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::manifest_communication_realization_attempts(),
            0
        );
        assert_eq!(
            super::super::path_instrumentation::snapshot(),
            Default::default()
        );
    }

    #[test]
    fn public_preparation_rejects_every_oversubscribed_pipeline_rank_before_native_work() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let backend = MlxBackend::new(execution.stream(), execution.stream());

        for rank in 0..3 {
            let topology = crate::test_parallel_rank(rank, 1, 3, 1);
            let options = crate::MlxLoadRequest::with_parallel(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            )
            .unwrap();
            super::super::path_instrumentation::reset();

            let error = backend
                .select_preparation(&inspection, &options)
                .unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains("decoder execution group 0 has 2 units for 3 pipeline stages"),
                "rank {rank} returned an unexpected selection failure: {error}"
            );
            assert_eq!(
                super::super::path_instrumentation::communication_realization_attempts(),
                0,
                "rank {rank} reached native communication realization"
            );
            assert_eq!(
                super::super::path_instrumentation::manifest_communication_realization_attempts(),
                0,
                "rank {rank} reached opaque manifest realization"
            );
            assert_eq!(
                super::super::path_instrumentation::snapshot(),
                Default::default(),
                "rank {rank} performed payload, construction, state, or execution work"
            );
        }
    }

    #[test]
    fn gguf_llama_tp_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();

        let selected = super::select_preparation(&inspection, options).unwrap();
        let manifest = selected
            .neutral()
            .communication_manifest()
            .expect("prediction-free Llama TP must retain neutral communication");
        assert_eq!(manifest.world_size(), 2);
        assert_eq!(manifest.rank(), 0);
        assert!(!manifest.groups().is_empty());
        assert!(selected.rank_context().is_some());
    }

    #[test]
    fn gguf_llama_pp_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 1, 2, 1);
        let options = crate::MlxLoadRequest::with_parallel(
            topology,
            crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            1,
            crate::MlxLoadRequest::test_communication_completion_policy(),
        )
        .unwrap();
        let selected = super::select_preparation(&inspection, options).unwrap();
        let manifest = selected
            .neutral()
            .communication_manifest()
            .expect("prediction-free Llama PP must retain neutral communication");
        assert_eq!(manifest.world_size(), 2);
        assert!(!manifest.routes().is_empty());
        let session_group = selected
            .neutral()
            .partitioned_session_group()
            .expect("PP admission must select one session-wide publication group");
        let descriptor = manifest
            .groups()
            .iter()
            .find(|group| group.id() == session_group)
            .expect("selected session group must be present in the manifest");
        assert_eq!(descriptor.creation_order(), 0);
        assert_eq!(descriptor.members(), [0, 1]);
        assert_eq!(descriptor.local_index(), Some(0));
        assert_eq!(
            descriptor
                .requirements()
                .operations()
                .iter()
                .map(eredu_runtime::CommunicationOperationRequirement::operation)
                .collect::<Vec<_>>(),
            [
                eredu_runtime::CommunicationOperation::Broadcast,
                eredu_runtime::CommunicationOperation::FailureAgreement,
            ]
        );
        let broadcast = &descriptor.requirements().operations()[0];
        assert_eq!(
            broadcast.dtypes(),
            [eredu_core::checkpoint::TensorDtype::F32]
        );
        assert!(broadcast.exact_completion());
        assert_eq!(broadcast.limits().unwrap().max_tensors(), 1);
        assert_eq!(broadcast.limits().unwrap().max_tensor_rank(), 3);
        assert!(descriptor.requirements().operations()[1].limits().is_none());
        assert!(selected.rank_context().is_some());
        assert!(selected.neutral().communication_manifest().is_some());
    }

    #[test]
    fn gguf_llama_bounded_residency_selects_neutral_partitioned_execution() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let layerwise = eredu_runtime::WeightResidency::layerwise_host(
            eredu_runtime::LayerwiseLoadOptions::new(OffloadConfig::new(None, None, 1).unwrap()),
        );
        let dense = eredu_runtime::WeightResidency::dense_disk_stream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
        );

        for residency in [layerwise, dense] {
            let topology = crate::test_parallel_rank(0, 2, 1, 1);
            let options = crate::MlxLoadRequest::with_parallel(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            )
            .unwrap()
            .with_weight_residency(residency);

            let selected = super::select_preparation(&inspection, options).unwrap();
            assert!(selected.neutral().communication_manifest().is_some());
            assert!(selected.rank_context().is_some());
        }
    }

    #[test]
    fn gguf_llama_tp_transform_selects_the_immutable_neutral_route() {
        let root = tempfile::tempdir().unwrap();
        let model = root.path().join("model.gguf");
        write_minimal_llama_gguf(&model, GgmlType::F16);
        let inspection = eredu_architectures::configuration::inspect_artifact(&model).unwrap();
        let topology = crate::test_parallel_rank(0, 2, 1, 1);
        let options =
            crate::MlxLoadRequest::with_quantization(eredu_core::QuantizationRequest::Affine {
                group_size: 16,
                bits: 4,
            })
            .with_parallel_topology(
                topology,
                crate::backend::DeviceAssignment::new(safemlx::DeviceType::Cpu, 0),
                eredu_runtime::PipelineWireContract::new(
                    eredu_runtime::PipelineActivationDtype::Float32,
                ),
                1,
                1,
                crate::MlxLoadRequest::test_communication_completion_policy(),
            )
            .unwrap();
        super::super::path_instrumentation::reset();

        let selected = super::select_preparation(&inspection, options).unwrap();
        assert!(selected.neutral().communication_manifest().is_some());
        assert!(selected.rank_context().is_some());
    }

    #[cfg(feature = "image")]
    #[test]
    fn mlx_processor_consumes_retained_qwen_plan_after_sidecar_removal() {
        use eredu_core::{ArtifactFormat, ModelConfigurationResolver};

        let root = tempfile::tempdir().unwrap();
        let sidecar = root
            .path()
            .join(eredu_architectures::processor_plan::PROCESSOR_CONFIG_FILENAME);
        std::fs::write(
            &sidecar,
            br#"{
                "size":{"shortest_edge":16,"longest_edge":64},
                "patch_size":2,"temporal_patch_size":2,"merge_size":2,
                "image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]
            }"#,
        )
        .unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_vl", "image_token_id": 61, "video_token_id": 62,
            "vision_start_token_id": 44, "vision_end_token_id": 45,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_vl_text", "hidden_size": 32,
                "num_hidden_layers": 3, "intermediate_size": 64,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
                "rms_norm_eps": 0.000001, "vocab_size": 64,
                "max_position_embeddings": 128, "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4, "hidden_size": 16, "intermediate_size": 24,
                "num_heads": 4, "num_position_embeddings": 16, "in_channels": 3,
                "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "out_hidden_size": 32, "deepstack_visual_indexes": [1, 3]
            }
        });
        let (configuration, resolved_plan) =
            eredu_architectures::configuration::MODEL_CONFIGURATIONS
                .resolve_safetensors(&config)
                .unwrap()
                .into_parts();
        let checkpoint = resolved_plan
            .safetensors_architecture()
            .unwrap()
            .checkpoint();
        let catalog = eredu_core::checkpoint::TensorCatalog::new(
            checkpoint
                .common_tensors
                .iter()
                .chain(
                    checkpoint
                        .layout_groups
                        .iter()
                        .filter(|group| group.required)
                        .filter_map(|group| group.variants.first())
                        .flat_map(|variant| variant.tensors.iter()),
                )
                .filter(|tensor| {
                    tensor.requirement == eredu_checkpoint::schema::TensorRequirement::Required
                })
                .map(|tensor| eredu_core::checkpoint::TensorDescriptor {
                    name: tensor.key.clone(),
                    shape: tensor.shape.clone(),
                    dtype: eredu_core::checkpoint::TensorDtype::F32,
                    storage: None,
                }),
        )
        .unwrap();
        let architecture_plan = eredu_architectures::configuration::MODEL_CONFIGURATIONS
            .artifact_plan(
                root.path(),
                ArtifactFormat::SafeTensors,
                &configuration,
                &catalog,
                None,
                resolved_plan,
            )
            .unwrap();
        std::fs::remove_file(sidecar).unwrap();

        assert_eq!(
            architecture_plan.model_kind(),
            eredu_architectures::ModelKind::Qwen3Vl
        );
        assert!(crate::composition::mlx::ModelProcessor::from_plan(&architecture_plan).is_some());
    }

    #[test]
    fn resolved_floating_dtype_selects_runtime_state_width() {
        for (dtype, bytes) in [
            (TensorDtype::F16, 2),
            (TensorDtype::Bf16, 2),
            (TensorDtype::F32, 4),
            (TensorDtype::F64, 8),
        ] {
            assert_eq!(mlx_floating_state_dtype_bytes(&dtype).unwrap().get(), bytes);
        }
    }

    #[test]
    fn dense_half_gguf_embeddings_select_two_byte_runtime_state() {
        for dtype in [GgmlType::F16, GgmlType::Bf16] {
            let root = tempfile::tempdir().unwrap();
            let path = root.path().join("model.gguf");
            write_minimal_llama_gguf(&path, dtype);
            let inspection = eredu_core::inspect_artifact(
                &path,
                &eredu_architectures::configuration::MODEL_CONFIGURATIONS,
            )
            .unwrap();

            assert_eq!(
                inspected_floating_state_dtype_bytes(&inspection)
                    .unwrap()
                    .get(),
                2
            );
        }
    }

    #[test]
    fn packed_embedding_dtype_uses_known_mlx_materialization_width() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::Encoded("F8_E4M3".into()))
                .unwrap()
                .get(),
            4
        );
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U32)
                .unwrap()
                .get(),
            4
        );
    }

    #[test]
    fn invalid_activation_dtype_does_not_silently_default() {
        assert_eq!(
            mlx_floating_state_dtype_bytes(&TensorDtype::U8),
            Err(TensorDtype::U8)
        );
    }
}

#[cfg(test)]
pub(super) fn prepared_safetensors_architecture(
    plan: &ArtifactArchitecturePlan,
) -> Result<&eredu_architectures::configuration::SafetensorsArchitecturePlan, Error> {
    plan.safetensors_architecture().ok_or_else(|| {
        Error::ArchitectureModel(
            "SafeTensors preparation omitted its validated architecture plan".into(),
        )
    })
}

fn attach_selected_processor(
    model: MlxModel,
    architecture_plan: &ArtifactArchitecturePlan,
    selected: &eredu_runtime::SelectedProcessorExecution,
) -> Result<MlxModel, Error> {
    #[cfg(any(feature = "image", feature = "audio"))]
    {
        Ok(model.with_processor(ModelProcessor::from_selected(architecture_plan, selected)?))
    }
    #[cfg(not(any(feature = "image", feature = "audio")))]
    {
        if selected.raw_media() {
            return Err(Error::ArchitectureModel(
                "selected raw-media execution has no compiled MLX mechanisms".into(),
            ));
        }
        let _ = architecture_plan;
        Ok(model)
    }
}
