//! Exact current external target sources enter the common native recipe compiler.
use super::source_bindings::SourceBindings;
use super::target_quote::completion::CompletionTrace;
use super::target_sources::ProjectedTargetEquationSources;
use crate::{
    backend::{
        error::Error,
        nn::{
            shared::MlxNeuralBackend,
            workspace::{
                ExistingArrayProjection, ProjectedNativeStorage, ProjectedResidentState,
                ResidentRecipeRecorder,
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
        speculative::{
            embedded_native::ExternalEquationRecipe, OriginalSpeculativeNumericalSources,
        },
    },
    MlxTensor,
};
use eredu_architectures::composite_execution::ExternalPredictionCaptureRequest;
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceDtype, HostMetadataFunding, WorkspaceTensor},
    Parameterized, Tensor,
};
use eredu_runtime::{
    speculative::external_occurrence::ExternalInvocation,
    working_memory::{InferenceWorkspaceReport, WorkingMemoryError},
    LayeredArchitecture, ReplicatedTextExecutionStrategy, ReplicatedTextSession,
};
use std::mem::{size_of, size_of_val};

pub(in crate::composition::mlx::replicated_text) struct ExternalTargetEquationQuote {
    pub report: InferenceWorkspaceReport,
    pub recipe: ExternalEquationRecipe,
    pub completion_roots: usize,
    pub state: ProjectedResidentState,
    pub inputs: ProjectedNativeStorage,
    pub layerwise: Option<LayerwiseWorkspace>,
    pub bindings: SourceBindings,
    pub context: WorkspaceContext,
    pub funding: HostMetadataFunding,
}
impl ExternalTargetEquationQuote {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::composition::mlx::replicated_text) fn inspect<A, S, D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        tokens: &MlxTensor,
        prefill: Option<&crate::composition::mlx::prepared_speculative::OriginalEmbeddedPrefillInput>,
        invocation: ExternalInvocation,
        capture: Option<&ExternalPredictionCaptureRequest>,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
        bind_sources: impl FnOnce(
            &WorkspaceContext,
            &[&ProjectedNativeStorage],
            Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
        ) -> Result<SourceBindings, Error>,
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
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ProjectedTargetEquationSources>(),
            size_of::<Result<ProjectedTargetEquationSources, Error>>(),
            size_of::<ExistingArrayProjection<'_>>(),
            size_of::<WorkspaceTensor>(),
            size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
            size_of::<ProjectedNativeStorage>(),
            size_of::<Result<ProjectedNativeStorage, eredu_nn::Error>>(),
            size_of::<SourceBindings>(),
            size_of::<Result<SourceBindings, Error>>(),
            size_of::<ResidentRecipeRecorder>(),
            size_of::<Result<ResidentRecipeRecorder, eredu_nn::Error>>(),
            size_of::<[i32; 2]>(),
            size_of::<[&ProjectedNativeStorage; 2]>(),
            size_of_val(&bind_sources),
            size_of::<(
                &MlxTensor,
                ExternalInvocation,
                &ExternalPredictionCaptureRequest,
                &OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>,
                &HostMetadataFunding,
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
        let geometry = invocation.geometry();
        let ProjectedTargetEquationSources {
            target,
            layerwise,
            context,
            batch,
            mechanism,
            addressable,
        } = ProjectedTargetEquationSources::inspect_geometry(
            session,
            geometry,
            sources,
            environment,
            funding,
        )?;
        let mut projection = ExistingArrayProjection::with_source_count(&context, 1)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let input = projection
            .project(tokens.as_array())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let shape = [
            i32::try_from(batch.get()).map_err(|cause| sources.retain_startup_error(cause))?,
            i32::try_from(geometry.input_positions)
                .map_err(|cause| sources.retain_startup_error(cause))?,
        ];
        use eredu_runtime::speculative::external_occurrence::ExternalInvocationKind as Kind;
        let valid=match invocation.kind(){
            Kind::TargetPrefill|Kind::TargetVerification|Kind::TargetTokenEmbeddings =>
                input.shape()==shape && matches!(input.layout().dtype(),WorkspaceDtype::Int32|WorkspaceDtype::Uint32),
            Kind::TargetProjectLogits =>input.shape().len()==3 && input.shape()[..2]==shape
                && input.layout().dtype()==WorkspaceDtype::Float32,
            _=>false,
        };
        if !valid || capture.is_some()!=matches!(invocation.kind(),Kind::TargetPrefill|Kind::TargetVerification){
            return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch));
        }
        let inputs = projection
            .try_into_storage()
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if !inputs.is_complete() {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        let paths=if capture.is_some(){
        let paths = session
            .shared_observation_paths()
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
        let prepared = session
            .prepared_observation_paths()
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
        let valid = session
            .inspect_runtime_execution_fixed(|_, _, runtime| {
                Ok::<_, std::convert::Infallible>(D::validate_observation_paths(runtime, prepared))
            })
            .map_err(|cause| sources.retain_startup_error(cause))?
            .unwrap_or_else(|never| match never {});
        valid.map_err(|cause| sources.retain_startup_error(cause))?;
            Some(paths)
        }else{None};
        let media = prefill.map(|source| source.project_media(&context, sources)).transpose()?.flatten();
        let bindings = bind_sources(&context, &[&target.storage, &inputs], media.as_ref().map(|input| input.source_storage()))?;
        let mut recorder = mechanism.recorder(geometry, &context)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        if let Some(source)=addressable {
            recorder.bind_addressable_sources(source)
                .map_err(|cause|sources.retain_startup_error(cause))?;
        }
        let mut completion = CompletionTrace::new(&mut recorder, &context)?;
        let report=match capture {
            Some(capture)=>sources.target_blueprint().quote_external_target_invocation(
                invocation,capture,paths.expect("capture binds paths"),&input,media,
                &target.state,&context,None,&mut completion),
            None=>{
                use eredu_architectures::composite_execution::ExternalPredictionTargetOperation as Operation;
                let operation=match invocation.kind(){
                    Kind::TargetTokenEmbeddings=>Operation::TokenEmbeddings(&input),
                    Kind::TargetProjectLogits=>Operation::ProjectLogits(&input),
                    _=>return Err(sources.retain_startup_error(WorkingMemoryError::IdentityMismatch)),
                };
                sources.target_blueprint().quote_external_target_static(invocation,operation,
                    &target.state,&context,None,&mut completion)
            }
        }.map_err(|cause|sources.retain_startup_error(cause))?;
        let completion_roots = completion.finish()?;
        let recipe = recorder
            .finish_external_operation(report.span_workspace_plan())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let recipe = ExternalEquationRecipe::new(invocation, recipe)?;
        Ok(Self {
            report,
            recipe,
            completion_roots,
            state: target,
            inputs,
            layerwise,
            bindings,
            context,
            funding: funding.clone(),
        })
    }
}
