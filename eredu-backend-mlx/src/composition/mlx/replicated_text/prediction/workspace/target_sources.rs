//! Shared immutable target projection for target and prediction invocations.
use crate::{
    backend::{
        error::Error,
        nn::{shared::MlxNeuralBackend, workspace::{ProjectedResidentState, ResidentExecutionMechanisms}},
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
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
    Parameterized,
};
use eredu_runtime::{
    replicated_session::RuntimeInspectionBoundary,
    speculative::embedded_occurrence::EmbeddedInvocationWorkspace,
    working_memory::WorkingMemoryError, LayeredArchitecture, ReplicatedTextExecutionStrategy,
    ReplicatedTextSession,
};
use std::{
    convert::Infallible,
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};

/// Owned native witnesses projected from the actual session, with the same
/// selected parameter representation and exact metadata context. This is not
/// execution authority; the caller retains its session loan until binding.
pub(super) struct ProjectedTargetEquationSources {
    pub target: ProjectedResidentState,
    pub layerwise: Option<LayerwiseWorkspace>,
    pub context: WorkspaceContext,
    pub batch: NonZeroU32,
    pub mechanism: ResidentExecutionMechanisms,
    pub addressable: Option<crate::backend::nn::workspace::AddressableSources>,
}
impl ProjectedTargetEquationSources {
    pub(super) fn inspect<A, S, D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        workspace: EmbeddedInvocationWorkspace,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
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
        Self::inspect_geometry(session, workspace.geometry(), sources, environment, funding)
    }
    pub(super) fn inspect_geometry<A, S, D>(
        session: &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
        geometry: eredu_core::InferenceGeometry,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
        funding: &HostMetadataFunding,
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
            size_of::<(&D, bool)>(),
            size_of::<Result<Self, Error>>(),
            size_of::<WorkspaceContext>(),
            size_of::<ResidentExecutionMechanisms>(),
            size_of::<Result<ResidentExecutionMechanisms, eredu_nn::Error>>(),
            size_of::<Result<WorkspaceContext, WorkspaceMetadataError>>(),
            size_of::<
                Result<
                    (&MlxReplicatedTextMechanisms<A, S>, &S, &D::Runtime),
                    RuntimeInspectionBoundary,
                >,
            >(),
            size_of::<Result<(&MlxReplicatedTextMechanisms<A, S>, &S, &D::Runtime), Infallible>>(),
            size_of::<Option<&MlxArchitectureLayerwisePolicy<A, S>>>(),
            size_of::<Option<&A::StaticModules>>(),
            size_of::<ProjectedResidentState>(),
            size_of::<Result<ProjectedResidentState, Error>>(),
            size_of::<Option<LayerwiseWorkspace>>(),
            size_of::<Result<LayerwiseWorkspace, Error>>(),
            size_of::<Result<NonZeroU32, Error>>(),
            size_of::<(
                &ReplicatedTextSession<A, MlxNeuralBackend, MlxReplicatedTextMechanisms<A, S>, D>,
                eredu_core::InferenceGeometry,
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
        sources.validate_environment(environment)?;
        sources
            .request()
            .validate_execution(session.inference_execution_identity())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        session
            .validate_control_state_origin_fixed(sources.target_origin())
            .map_err(|cause| sources.retain_startup_error(cause))?;
        // Fixed inspection only lends immutable sources. Costly projection
        // happens after returning from the boundary callback.
        let (_, target_state, runtime) = session
            .inspect_runtime_execution_fixed(|m, s, d| Ok::<_, Infallible>((m, s, d)))
            .map_err(|cause| sources.retain_startup_error(cause))?
            .unwrap_or_else(|never| match never {});
        let (_, ordinary) = sources.numerical_prerequisites();
        // Carry the same descriptive choice to every consumer. The retained
        // environment remains borrowed; no stream or mechanism is selected here.
        let mechanism = ResidentExecutionMechanisms::from_stream(ordinary, environment.stream(), funding)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        let addressable=sources.target_addressable_sources(mechanism,environment,funding)?;
        if !session.execution_strategy().uses_ordinary_unit_equations() && addressable.is_none() {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        let context = match &addressable {
            Some(source)=>WorkspaceContext::new_with_metadata_funding(
                crate::backend::nn::workspace::MlxAddressableWorkspaceMechanisms::new(mechanism,source.clone()),funding.clone()),
            None=>mechanism.context(funding.clone()),
        }.map_err(|cause| sources.retain_startup_error(cause))?;
        let selected = D::resident_policy(runtime)
            .or_else(|| D::bounded_policy(runtime))
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        let modules = D::static_modules_ref(runtime)
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::UnknownBound))?;
        selected
            .install_workspace_parameter_representations(modules, &context)
            .map_err(|cause| sources.retain_startup_error(cause))?;
        // The selected policy also supplies resident parameter facts. Only the
        // retained host/disk realization needs a separate materialization plan.
        let layerwise = match sources.target_blueprint().selected().text_realization().residency() {
            eredu_runtime::LayerWeightResidency::FullyResident => None,
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
            | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => Some(
                selected
                    .prepared_layerwise_workspace(ordinary.allocation(), &context)
                    .map_err(|cause| sources.retain_error(cause))?,
            ),
            _ => return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound)),
        };
        let batch = u32::try_from(geometry.batch_size)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| sources.retain_startup_error(WorkingMemoryError::IdentityMismatch))?;
        let target = target_state
            .project_resident_workspace_with_storage(batch, &context)
            .map_err(|cause| sources.retain_error(cause))?;
        if !target.storage.is_complete() {
            return Err(sources.retain_startup_error(WorkingMemoryError::UnknownBound));
        }
        Ok(Self {
            target,
            layerwise,
            context,
            batch,
            mechanism,
            addressable,
        })
    }
}
