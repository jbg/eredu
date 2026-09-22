//! Exact native sources retained through the shared prediction equation quote.
use super::super::{
    prepare_prediction_parameters, MlxEmbeddedPredictionMaterializer,
    MlxWorkspacePredictionParameterSource, NativePredictionParameters, PredictionParameterStorage,
};
use super::source_bindings::SourceBindings;
use super::{project_prediction_state, ProjectedPredictionState};
use crate::{
    backend::{
        error::Error,
        nn::{
            shared::MlxNeuralBackend,
            workspace::{
                EmbeddedEquationRecipe, ExistingArrayProjection, ProjectedNativeStorage,
                ProjectedResidentState, ResidentRecipeRecorder,
            },
        },
        runtime::execution::generic::LayerwiseWorkspace,
        OriginalCopyEnvironment,
    },
    composition::mlx::{
        replicated_text::{
            session::MlxReplicatedTextMechanisms, MlxArchitectureLayerwisePolicy,
            MlxStateMechanisms,
        },
        speculative::OriginalSpeculativeNumericalSources,
    },
    MlxTensor,
};
use eredu_architectures::prediction_extension::{
    equation::PredictionEquation, workspace::WorkspacePredictionState,
    MaterializedPredictionExecutor,
};
use eredu_nn::{
    workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor},
    Parameterized,
};
use eredu_runtime::{
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::{InferenceWorkspaceReport, WorkingMemoryError},
    LayeredArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
};
use std::{
    marker::PhantomData,
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};
mod io;
pub(in crate::composition::mlx::replicated_text) use io::{PredictionIoPlan, PreparedPredictionIo};

/// Immutable current session/lane/input loans remain live until native binding
/// consumes this quote. The report itself grants no mutation or source authority.
pub(in crate::composition::mlx::replicated_text) struct PreparedPredictionEquationQuote<'source> {
    parts: PredictionEquationQuoteParts,
    sources: &'source OriginalSpeculativeNumericalSources,
    loan: PhantomData<&'source ()>,
}
/// Exact owned source witnesses survive releasing the immutable execution loan.
/// Native registration must consume these before beginning any mutable equation.
pub(in crate::composition::mlx::replicated_text) struct PredictionEquationQuoteParts {
    pub report: InferenceWorkspaceReport,
    pub capture: crate::backend::array_copy::CaptureNativePopulation,
    pub recipe: EmbeddedEquationRecipe,
    pub target: ProjectedResidentState,
    pub prediction: ProjectedNativeStorage,
    pub inputs: ProjectedNativeStorage,
    pub parameters: PredictionParameterStorage,
    pub layerwise: Option<LayerwiseWorkspace>,
    pub io: PredictionIoPlan,
    pub context: WorkspaceContext,
    pub bindings: SourceBindings,
    // Inspection clones, reports, containers and their Context retire first.
    pub funding: HostMetadataFunding,
}
impl<'source> PreparedPredictionEquationQuote<'source> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx::replicated_text) fn inspect<A, S, D, P>(
        session: &'source ReplicatedTextSession<
            A,
            MlxNeuralBackend,
            MlxReplicatedTextMechanisms<A, S>,
            D,
        >,
        prediction: &'source P,
        lane: &'source P::LaneState,
        equation: PredictionEquation<&'source MlxTensor>,
        workspace: EmbeddedInvocationWorkspace,
        sources: &'source OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
        observation: Option<super::capture::CaptureWorkspaceInput<'_, '_>>,
        bind_sources: impl FnOnce(
            &WorkspaceContext,
            &[&ProjectedNativeStorage],
        ) -> Result<SourceBindings, Error>,
    ) -> Result<Self, Error>
    where
        S: MlxStateMechanisms,
        A: LayeredArchitecture<MlxNeuralBackend, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Parameterized<MlxTensor>,
        A::Unit: Parameterized<MlxTensor> + 'static,
        D: ReplicatedTextExecutionStrategy<
            A,
            MlxNeuralBackend,
            S,
            MlxArchitectureLayerwisePolicy<A, S>,
            MlxArchitectureLayerwisePolicy<A, S>,
        >,
        P: MaterializedPredictionExecutor<A, MlxNeuralBackend, MlxEmbeddedPredictionMaterializer>
            + 'static,
    {
        let controls = [
            size_of_val(&bind_sources),
            size_of::<&HostMetadataFunding>(),
            size_of::<[&ProjectedNativeStorage; 4]>(),
            size_of::<Option<SourceBindings>>(),
            size_of::<Result<SourceBindings, Error>>(),
            size_of::<Self>(),
            size_of::<std::cell::Cell<crate::backend::array_copy::CaptureNativePopulation>>(),
            size_of::<crate::backend::array_copy::CaptureNativePopulation>(),
            size_of::<Option<u64>>(),
            size_of::<PredictionEquationQuoteParts>(),
            size_of::<Result<Self, Error>>(),
            size_of::<PredictionEquation<&MlxTensor>>(),
            size_of::<PredictionEquation<WorkspaceTensor>>(),
            size_of::<Result<PredictionEquation<WorkspaceTensor>, eredu_nn::Error>>(),
            size_of::<EmbeddedInvocationWorkspace>(),
            size_of::<Option<super::capture::CaptureWorkspaceInput<'_, '_>>>(),
            size_of::<Option<ProjectedNativeStorage>>(),
            size_of::<super::ProjectedPredictionLane>(),
            size_of::<Result<super::ProjectedPredictionLane, eredu_nn::Error>>(),
            size_of::<ExistingArrayProjection<'_>>(),
            size_of::<Result<ProjectedNativeStorage, eredu_nn::Error>>(),
            size_of::<(
                NativePredictionParameters<'_, A, P>,
                PredictionParameterStorage,
            )>(),
            size_of::<
                Result<
                    (
                        NativePredictionParameters<'_, A, P>,
                        PredictionParameterStorage,
                    ),
                    eredu_nn::Error,
                >,
            >(),
            size_of::<Option<ResidentRecipeRecorder>>(),
            size_of::<Option<crate::backend::nn::workspace::ParallelRecipeRecorder>>(),
            size_of::<
                Result<
                    Option<crate::backend::nn::workspace::ParallelRecipeRecorder>,
                    eredu_nn::Error,
                >,
            >(),
            size_of::<&mut dyn crate::composition::mlx::model::CaptureRecorder>(),
            size_of::<Option<crate::composition::mlx::model::NativeLayerwiseParameters<'_>>>(),
            size_of::<
                Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
            >(),
            size_of::<Option<&eredu_runtime::RetainedCommunicationSource>>(),
            size_of::<Result<ResidentRecipeRecorder, eredu_nn::Error>>(),
            size_of::<
                Result<
                    InferenceWorkspaceReport,
                    eredu_architectures::prepared_execution::PreparedExecutionError<
                        eredu_nn::Error,
                    >,
                >,
            >(),
            size_of::<Result<EmbeddedEquationRecipe, eredu_nn::Error>>(),
            size_of::<(
                &P::LaneState,
                &mut Option<ProjectedNativeStorage>,
                NonZeroU32,
            )>(),
            OriginalCopyEnvironment::control_bytes()
                .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::Overflow))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let super::target_sources::ProjectedTargetEquationSources {
            target,
            layerwise,
            context,
            batch,
            mechanism,
            addressable,
            parallel,
        } = super::target_sources::ProjectedTargetEquationSources::inspect(
            session,
            workspace,
            sources,
            environment,
            funding,
        )?;
        let runtime = environment
            .input_runtime()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let mut io = PredictionIoPlan::inspect(&equation, workspace, &runtime, &context)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let count = match &equation {
            PredictionEquation::Prefill { .. } => 3,
            PredictionEquation::Sequential { .. } => 1,
            PredictionEquation::Fused { .. } => 0,
            PredictionEquation::Replay { .. } => 2,
        };
        let mut input_projection = ExistingArrayProjection::with_source_count(&context, count)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let equation = equation
            .try_map(|value| input_projection.project(value.as_array()))
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let inputs = input_projection
            .try_into_storage()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let (parameters, parameter_storage) =
            prepare_prediction_parameters::<A, P>(prediction, layerwise.as_ref(), &context)
                .map_err(|cause| sources.retain_startup_error(cause))?;
        if !target.storage.is_complete()
            || !inputs.is_complete()
            || !parameter_storage.storage().is_complete()
        {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        let mut prediction_storage = None;
        let mut source_bindings = None;
        let mut local_recorder = if parallel.is_none() {
            let mut recorder = mechanism
                .recorder(workspace.geometry(), &context)
                .map_err(|cause| sources.retain_startup_error(cause))?;
            if let Some(source) = addressable {
                recorder
                    .bind_addressable_sources(source)
                    .map_err(|cause| sources.retain_startup_error(cause))?;
            }
            Some(recorder)
        } else {
            None
        };
        let mut parallel_recorder = parallel
            .as_ref()
            .map(|source| source.recorder(workspace.geometry()))
            .transpose()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let recorder: &mut dyn crate::composition::mlx::model::CaptureRecorder =
            match (local_recorder.as_mut(), parallel_recorder.as_mut()) {
                (Some(recorder), None) => recorder,
                (None, Some(recorder)) => recorder,
                _ => return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
            };

        let transfers =
            std::cell::Cell::new(crate::backend::array_copy::CaptureNativePopulation::default());
        let logical_prediction = observation
            .as_ref()
            .map(|source| source.invocation.origin().prediction as u64);
        let mut capture_observer = observation
            .map(|source| source.prepare(workspace, &context, &transfers, sources))
            .transpose()?;
        let observation = capture_observer.as_mut().zip(logical_prediction).map(|(observer, prediction)|
            eredu_architectures::prepared_execution::EmbeddedPredictionWorkspaceObservation::new(observer, prediction));
        let target_parameters = parallel
            .as_ref()
            .and(layerwise.as_ref())
            .map(crate::composition::mlx::model::NativeLayerwiseParameters);
        let target_parameters = target_parameters.as_ref().map(|parameters| {
            parameters as &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
        });
        let communication = parallel.as_ref().map(|source| source.declaration_source());
        // Exact unloaded slots/copies are bound separately from numerical Eval;
        // no ParameterPlaceholder operation is imported into the equation trace.
        let report = sources
            .target_blueprint()
            .quote_embedded_prediction_invocation::<MlxWorkspacePredictionParameterSource<A, P>, _>(
                workspace,
                equation,
                &target.state,
                &context,
                target_parameters,
                communication,
                parameters,
                |layout, context| {
                    let projected = project_prediction_state::<A, P>(lane, layout, batch, context)?;
                    if !projected.storage.is_complete() {
                        return Err(context.metadata_source(WorkingMemoryError::UnknownBound));
                    }
                    source_bindings = Some(
                        bind_sources(
                            context,
                            &[
                                &target.storage,
                                &projected.storage,
                                &inputs,
                                parameter_storage.storage(),
                            ],
                        )
                        .map_err(|cause| context.metadata_source(cause))?,
                    );
                    prediction_storage = Some(projected.storage);
                    Ok(match projected.state {
                        ProjectedPredictionState::Sequential(state) => {
                            WorkspacePredictionState::Sequential(state)
                        }
                        ProjectedPredictionState::Pooling(state) => {
                            WorkspacePredictionState::Pooling(state)
                        }
                        ProjectedPredictionState::Model(state) => {
                            WorkspacePredictionState::Model(state)
                        }
                    })
                },
                observation,
                &mut io,
                recorder,
            )
            .map_err(|cause| sources.retain_startup_error(cause))?;
        io.finish_quote()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let capture = transfers.get();
        recorder
            .capture_population(capture)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        drop(capture_observer);
        let recipe = match (local_recorder, parallel_recorder) {
            (Some(recorder), None) => {
                recorder.finish_embedded(report.span_workspace_plan(), workspace)
            }
            (None, Some(recorder)) => {
                recorder.finish_embedded(report.span_workspace_plan(), workspace)
            }
            _ => return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
        }
        .map_err(|cause| sources.retain_startup_error(cause))?;
        let prediction_storage = prediction_storage
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        Ok(Self {
            parts: PredictionEquationQuoteParts {
                report,
                capture,
                recipe,
                target,
                prediction: prediction_storage,
                inputs,
                parameters: parameter_storage,
                layerwise,
                io,
                context,
                bindings: source_bindings.ok_or_else(|| {
                    sources.retain_startup_error(WorkingMemoryError::UnknownBound)
                })?,
                funding: funding.clone(),
            },
            sources,
            loan: PhantomData,
        })
    }
    pub(in crate::composition::mlx::replicated_text) fn parts(
        &self,
    ) -> &PredictionEquationQuoteParts {
        &self.parts
    }
    pub(in crate::composition::mlx::replicated_text) fn sources(
        &self,
    ) -> &OriginalSpeculativeNumericalSources {
        self.sources
    }
    pub(in crate::composition::mlx::replicated_text) fn into_parts(
        self,
    ) -> PredictionEquationQuoteParts {
        self.parts
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
