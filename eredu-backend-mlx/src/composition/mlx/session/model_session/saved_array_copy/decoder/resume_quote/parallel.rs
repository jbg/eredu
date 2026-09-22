//! The exact retained native world drives the copied local-state quote.
use super::*;
use crate::backend::nn::workspace::{
    AddressableSources, MlxAddressableWorkspaceMechanisms, MlxParallelWorkspace,
    MlxParallelWorkspaceMechanisms, ResidentExecutionMechanisms, ResidentNativeRecipe,
};
use crate::composition::mlx::model::retain_planning_error;
use crate::composition::mlx::session::intervention::TextInterventionQuote;

pub(super) enum SavedWorkspace {
    Resident {
        context: WorkspaceContext,
        addressable: Option<AddressableSources>,
    },
    Parallel(MlxParallelWorkspace),
}
impl SavedWorkspace {
    pub(super) fn prepare(
        runtime: &ModelRuntime<MlxBackend<'_>>,
        mechanisms: ResidentExecutionMechanisms,
        funding: Option<&HostMetadataFunding>,
    ) -> Result<Self, Error> {
        if let Some(funding) = funding {
            funding
                .reserve_metadata(std::mem::size_of::<(
                    Self,
                    Result<Self, Error>,
                    Option<&HostMetadataFunding>,
                    &ModelRuntime<MlxBackend<'_>>,
                )>())
                .map_err(Error::WorkspacePlanning)?;
        }
        let session = runtime.session();
        let addressable = funding
            .map(|funding| {
                session.payload.model.prepare_addressable_workspace_sources(
                    mechanisms,
                    &session.payload.memory_ledger,
                    funding,
                )
            })
            .transpose()?
            .flatten();
        match (&session.payload.distributed, funding) {
            (Some(_distributed), Some(funding)) => {
                funding.reserve_metadata(std::mem::size_of::<(
                    crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource,
                    MlxParallelWorkspaceMechanisms,
                )>()).map_err(Error::WorkspacePlanning)?;
                let result = (|| {
                    let source = session
                        .original_workspace_parallel_source(funding)?
                        .ok_or_else(unknown)?;
                    if !source.funding().same_account(funding) {
                        return Err(memory(WorkingMemoryError::IdentityMismatch));
                    }
                    let mechanism = MlxParallelWorkspaceMechanisms::new(mechanisms, source);
                    match &addressable {
                        Some(source) => {
                            mechanism.prepare_workspace_with_addressable(source.clone())
                        }
                        None => mechanism.prepare_workspace(),
                    }
                    .map(Self::Parallel)
                    .map_err(Error::Neural)
                })();
                result.map_err(|cause| retain_planning_error(cause, funding.clone()))
            }
            (Some(_), None) => Err(unknown()),
            (None, Some(funding)) => {
                let context = match &addressable {
                    Some(source) => WorkspaceContext::new_with_metadata_funding(
                        MlxAddressableWorkspaceMechanisms::new(mechanisms, source.clone()),
                        funding.clone(),
                    ),
                    None => {
                        WorkspaceContext::new_with_metadata_funding(mechanisms, funding.clone())
                    }
                }
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
                Ok(Self::Resident {
                    context,
                    addressable,
                })
            }
            (None, None) => Ok(Self::Resident {
                context: WorkspaceContext::new(mechanisms),
                addressable: None,
            }),
        }
    }
    pub(super) fn context(&self) -> &WorkspaceContext {
        match self {
            Self::Resident { context, .. } => context,
            Self::Parallel(parallel) => parallel.context(),
        }
    }
    pub(super) fn addressable(&self) -> Option<&AddressableSources> {
        match self {
            Self::Resident { addressable, .. } => addressable.as_ref(),
            Self::Parallel(value) => value.addressable_sources(),
        }
    }
    pub(super) fn parallel(&self) -> Option<&MlxParallelWorkspace> {
        match self {
            Self::Parallel(parallel) => Some(parallel),
            Self::Resident { .. } => None,
        }
    }
}

type QuoteParts = (
    PreparedTextGenerationWorkspace,
    Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ResidentNativeRecipe,
);

pub(super) fn quote(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    geometry: InferenceGeometry,
    state: &DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>,
    sampling: BorrowedTextSamplingWorkspace<'_>,
    parallel: &MlxParallelWorkspace,
    capture: Option<(
        &eredu_runtime::capture::FundedCaptureCheckpoint,
        &eredu_runtime::layered::PreparedCaptureSelection,
    )>,
    interventions: Option<TextInterventionQuote<'_>>,
    parameters: crate::composition::mlx::model::PreparedSavedParameterSource,
) -> Result<QuoteParts, Error> {
    let context = parallel.context();
    let funding = context.metadata_funding().ok_or_else(unknown)?;
    let result = (|| {
        context
            .charge_metadata(std::mem::size_of::<(
                Result<QuoteParts, Error>,
                Option<TextInterventionQuote<'_>>,
                BorrowedTextSamplingWorkspace<'_>,
                crate::composition::mlx::model::NativeLayerwiseParameters<'_>,
                Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                Option<(
                    &eredu_runtime::capture::FundedCaptureCheckpoint,
                    &eredu_runtime::layered::PreparedCaptureSelection,
                )>,
                (
                    &eredu_architectures::component_partition::ComponentPartitionLayouts,
                    usize,
                ),
            )>())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let model = &runtime.session().payload.model;
        let blueprint = model.inference_blueprint().ok_or_else(unknown)?;
        let layerwise = parameters.layerwise;
        let mut recorder = parallel.recorder(geometry).map_err(Error::Neural)?;
        if let Some(source) = &layerwise {
            recorder
                .bind_layerwise_span_constructor_source(source)
                .map_err(Error::Neural)?;
        }
        if capture.is_none() && interventions.is_some() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        let generation = if let Some((checkpoint, selection)) = capture {
            let session = runtime.session();
            let loaded = session.partition_capture_source().ok_or_else(unknown)?;
            let (_, execution, setup) = loaded.source_labels();
            let distributed = session.payload.distributed.as_ref().ok_or_else(unknown)?;
            let manifest = blueprint
                .selected()
                .communication_manifest()
                .ok_or_else(unknown)?;
            if setup != distributed.session_identity()
                || execution != blueprint.execution_identity()
            {
                return Err(memory(WorkingMemoryError::IdentityMismatch));
            }
            let parameters = layerwise
                .as_ref()
                .map(crate::composition::mlx::model::NativeLayerwiseParameters);
            model.quote_partitioned_saved_capture(
                blueprint,
                geometry,
                state,
                context,
                sampling,
                checkpoint,
                selection,
                interventions,
                &mut recorder,
                parallel.declaration_source(),
                (loaded.layouts(), manifest.rank()),
                parameters.as_ref().map(|source| {
                    source as
                    &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
                }),
            )
        } else {
            match layerwise.as_ref() {
                Some(source) => blueprint
                    .quote_partitioned_layerwise_text_with_existing_sampling_and_trace(
                        geometry,
                        state,
                        context,
                        sampling,
                        &mut recorder,
                        Some(parallel.declaration_source()),
                        &crate::composition::mlx::model::NativeLayerwiseParameters(source),
                    ),
                None => blueprint.quote_partitioned_text_with_existing_sampling_and_trace(
                    geometry,
                    state,
                    context,
                    sampling,
                    &mut recorder,
                    Some(parallel.declaration_source()),
                ),
            }
        }
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let mut recipe = recorder
            .finish(generation.equations.span_workspace_plan())
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        model.erased().bind_layerwise_neural_recipe(
            &runtime.session().payload.memory_ledger,
            &mut recipe,
            Some(&funding),
        )?;
        if let Some(source) = &layerwise {
            source.bind_native_host_copies(&mut recipe)?;
        }
        Ok((generation, layerwise, recipe))
    })();
    result.map_err(|cause: Error| retain_planning_error(cause, funding))
}
