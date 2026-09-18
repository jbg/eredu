//! Native source-bound target verification/prefill quote for embedded prediction.
pub(super) mod completion;
use super::source_bindings::SourceBindings;
use super::target_sources::ProjectedTargetEquationSources;
use crate::{
    MlxTensor,
    backend::{
        OriginalCopyEnvironment,
        error::Error,
        nn::{
            shared::MlxNeuralBackend,
            workspace::{
                EmbeddedEquationRecipe, ExistingArrayProjection, ProjectedNativeStorage,
                ProjectedResidentState, ResidentRecipeRecorder,
            },
        },
        runtime::execution::generic::LayerwiseWorkspace,
    },
    composition::mlx::{
        replicated_text::{
            MlxArchitectureLayerwisePolicy, MlxStateMechanisms,
            session::MlxReplicatedTextMechanisms,
        },
        speculative::OriginalSpeculativeNumericalSources,
    },
};
use eredu_architectures::prepared_execution::{
    EmbeddedTargetWorkspaceObservation, PreparedExecutionError,
};
use eredu_nn::{
    Parameterized, Tensor,
    workspace::{WorkspaceContext, WorkspaceDtype, HostMetadataFunding, WorkspaceTensor},
};
use eredu_runtime::{
    LayeredArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::{InferenceWorkspaceReport, WorkingMemoryError},
};
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

/// Immutable session/input loans are consumed before mutation; the owned parts
/// retain the actual native backing to bind into the same invocation.
pub(in crate::composition::mlx::replicated_text) struct PreparedTargetEquationQuote<'source> {
    parts: TargetEquationQuoteParts,
    sources: &'source OriginalSpeculativeNumericalSources,
    loan: PhantomData<&'source ()>,
}
pub(in crate::composition::mlx::replicated_text) struct TargetEquationQuoteParts {
    pub report: InferenceWorkspaceReport,
    pub capture: crate::backend::array_copy::CaptureNativePopulation,
    pub completion_roots: usize,
    pub recipe: EmbeddedEquationRecipe,
    pub target: ProjectedResidentState,
    pub inputs: ProjectedNativeStorage,
    pub layerwise: Option<LayerwiseWorkspace>,
    pub context: WorkspaceContext,
    pub bindings: SourceBindings,
    // Native projection witnesses and metadata containers retire first.
    pub funding: HostMetadataFunding,
}
impl<'source> PreparedTargetEquationQuote<'source> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx::replicated_text) fn inspect<A, S, D>(
        session: &'source ReplicatedTextSession<
            A,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<A, S>,
            D,
        >,
        tokens: &'source MlxTensor,
        prefill: Option<&crate::composition::mlx::prepared_speculative::OriginalEmbeddedPrefillInput>,
        workspace: EmbeddedInvocationWorkspace,
        sources: &'source OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
        observation: Option<super::capture::CaptureWorkspaceInput<'_, '_>>,
        bind_sources: impl FnOnce(&WorkspaceContext, &[&ProjectedNativeStorage], Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>) -> Result<SourceBindings, Error>,
    ) -> Result<Self, Error>
    where
        S: MlxStateMechanisms,
        A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error>,
        A::StaticModules: Parameterized<MlxTensor>,
        A::Unit: Parameterized<MlxTensor> + 'static,
        D: ReplicatedTextExecutionStrategy<
                A,
                MlxNeuralBackend,
                S,
                MlxArchitectureLayerwisePolicy<A, S>,
                MlxArchitectureLayerwisePolicy<A, S>,
            >,
    {
        let controls = [
            size_of_val(&bind_sources),
            size_of::<[&ProjectedNativeStorage; 2]>(),
            size_of::<SourceBindings>(),
            size_of::<Result<SourceBindings, Error>>(),
            size_of::<Self>(),
            size_of::<std::cell::Cell<crate::backend::array_copy::CaptureNativePopulation>>(),
            size_of::<crate::backend::array_copy::CaptureNativePopulation>(),
            size_of::<Option<u64>>(),
            size_of::<Option<&eredu_runtime::SharedLayeredObservationPaths>>(),
            size_of::<Option<&eredu_runtime::PreparedLayeredObservationPaths>>(),
            size_of::<
                Result<
                    (),
                    eredu_runtime::ReplicatedTextSessionError<
                        eredu_nn::Error,
                        <MlxArchitectureLayerwisePolicy<A, S> as eredu_runtime::LayerwisePolicy<
                            MlxNeuralBackend,
                            A::Unit,
                        >>::Error,
                        std::convert::Infallible,
                    >,
                >,
            >(),
            size_of::<TargetEquationQuoteParts>(),
            size_of::<Result<Self, Error>>(),
            size_of::<EmbeddedInvocationWorkspace>(),
            size_of::<Option<super::capture::CaptureWorkspaceInput<'_, '_>>>(),
            size_of::<ExistingArrayProjection<'_>>(),
            size_of::<WorkspaceTensor>(),
            size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
            size_of::<ProjectedNativeStorage>(),
            size_of::<Result<ProjectedNativeStorage, eredu_nn::Error>>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<Result<ResidentRecipeRecorder, eredu_nn::Error>>(),
            size_of::<Result<InferenceWorkspaceReport, PreparedExecutionError<eredu_nn::Error>>>(),
            size_of::<Result<EmbeddedEquationRecipe, eredu_nn::Error>>(),
            size_of::<[i32; 2]>(),
            size_of::<(
                &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
                &MlxTensor,
                EmbeddedInvocationWorkspace,
                &OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>,
                &HostMetadataFunding,
                Option<super::capture::CaptureWorkspaceInput<'_, '_>>,
            )>(),
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let ProjectedTargetEquationSources {
            target,
            layerwise,
            context,
            batch,
            mechanism,
            addressable,
        } = ProjectedTargetEquationSources::inspect(session, workspace, sources, environment, funding)?;
        let mut projection = ExistingArrayProjection::with_source_count(&context, 1)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let input = projection
            .project(tokens.as_array())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let extent = [
            i32::try_from(batch.get()).map_err(|cause| sources.retain_startup_error(cause))?,
            i32::try_from(workspace.geometry().input_positions)
                .map_err(|cause| sources.retain_startup_error(cause))?,
        ];
        if input.shape() != extent
            || !matches!(
                input.layout().dtype(),
                WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
            )
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let inputs = projection
            .try_into_storage()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if !inputs.is_complete() {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        let media = prefill.map(|source| source.project_media(&context, sources)).transpose()?.flatten();
        let bindings = bind_sources(&context, &[&target.storage, &inputs], media.as_ref().map(|input| input.source_storage()))?;
        let mut recorder =
            mechanism.recorder(workspace.geometry(), &context)
                .map_err(|cause| sources.retain_startup_error(cause))?;
        if let Some(source)=addressable {
            recorder.bind_addressable_sources(source)
                .map_err(|cause|sources.retain_startup_error(cause))?;
        }
        let transfers =
            std::cell::Cell::new(crate::backend::array_copy::CaptureNativePopulation::default());
        let logical_prediction = observation
            .as_ref()
            .map(|source| source.invocation.origin().prediction as u64);
        let mut capture_observer = observation
            .map(|source| source.prepare(workspace, &context, &transfers, sources))
            .transpose()?;
        let paths = if capture_observer.is_some() {
            let paths = session.shared_observation_paths().ok_or_else(|| {
                sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)
            })?;
            let prepared = session.prepared_observation_paths().ok_or_else(|| {
                sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)
            })?;
            // Borrow the exact retained execution. This fixed inspector does not
            // build ordinary inspection diagnostics or a new path inventory.
            let valid = session
                .inspect_runtime_execution_fixed(|_, _, runtime| {
                    Ok::<_, std::convert::Infallible>(D::validate_observation_paths(
                        runtime, prepared,
                    ))
                })
                .map_err(|cause| sources.retain_startup_error(cause))?
                .unwrap_or_else(|never| match never {});
            valid.map_err(|cause| sources.retain_startup_error(cause))?;
            Some(paths)
        } else {
            None
        };
        let observation = capture_observer
            .as_mut()
            .zip(logical_prediction)
            .zip(paths)
            .map(|((observer, prediction), paths)| {
                EmbeddedTargetWorkspaceObservation::new(paths, observer, prediction)
            });
        // The common equation worker excludes its explicit token placeholder.
        // Native binding must consume this actual input storage separately; the
        // placeholder contributes no source allocation or submission authority.
        let mut completion = completion::CompletionTrace::new(&mut recorder, &context)?;
        let report = sources
            .target_blueprint()
            .quote_embedded_target_invocation(
                workspace,
                input.layout().dtype(),
                media,
                &target.state,
                &context,
                None,
                observation,
                &mut completion,
            )
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let completion_roots = completion.finish()?;
        let capture = transfers.get();
        recorder
            .record_capture_population(capture)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        drop(capture_observer);
        let recipe = recorder
            .finish_embedded(report.span_workspace_plan(), workspace)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        Ok(Self {
            parts: TargetEquationQuoteParts {
                report,
                capture,
                completion_roots,
                recipe,
                target,
                inputs,
                layerwise,
                context,
                bindings,
                funding: funding.clone(),
            },
            sources,
            loan: PhantomData,
        })
    }
    pub(in crate::composition::mlx::replicated_text) fn parts(&self) -> &TargetEquationQuoteParts {
        &self.parts
    }
    pub(in crate::composition::mlx::replicated_text) fn sources(
        &self,
    ) -> &OriginalSpeculativeNumericalSources {
        self.sources
    }
    pub(in crate::composition::mlx::replicated_text) fn into_parts(
        self,
    ) -> TargetEquationQuoteParts {
        self.parts
    }
}
